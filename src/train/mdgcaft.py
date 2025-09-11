#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reverse-CAFT (gradient projection ablation) trainer for Qwen2.5 Instruct models (32B works),
with first-class support for 1-D mean-diff directions.

What this does:
  • Load base model + tokenizer (HF Transformers)
  • Build JSONL dataset (naive "concat" formatting; plug your own if needed)
  • Infer sensible LoRA target modules for Qwen2.5 variants
  • Load per-layer vectors/subspaces from --em LAYER:FILE (1-D mean-diff or [d,k] PCA/SAE)
  • Orthonormalize columns (QR) unless --no_unit_norm is set
  • Attach Reverse-CAFT gradient projector hooks at the chosen layers
  • Optional activation-energy probe before training
  • Train with HF Trainer (bf16), then report gradient-removal stats and save the adapter

Example (mean-diff):
  python train_rev_caft_meandiff_qwen.py \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --dataset .caft/emergent_misalignment/datasets/insecure_subset.jsonl \
    --em 12:results/mean_diff/meandiff.pt \
    --em 32:results/mean_diff/meandiff.pt \
    --em 50:results/mean_diff/meandiff.pt \
    --out runs/qwen32b_revcaft_meandiff \
    --epochs 1 --lr 1e-5 --bsz 1 --accum 2 --seed 42

Minimal deps (no funky venv):
  pip install 'transformers>=4.41' 'peft>=0.10.0' accelerate torch
"""

import os
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model, TaskType
from peft.tuners.lora import LoraLayer


# =========================
# Dataset
# =========================

class JsonlCausalDataset(Dataset):
    """
    Accepts JSONL lines with one of:
      {"text": "..."}
      {"prompt": "...", "completion": "..."}          -> concatenated
      {"messages": [{"role": "...", "content": "..."} , ...]} -> naive join
    """
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_len: int = 2048):
        self.tok = tokenizer
        self.max_len = max_len
        self.samples: List[str] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = None
                if isinstance(obj, dict):
                    if "text" in obj and isinstance(obj["text"], str):
                        text = obj["text"]
                    elif "prompt" in obj and "completion" in obj:
                        text = str(obj["prompt"]) + str(obj["completion"])
                    elif "messages" in obj and isinstance(obj["messages"], list):
                        text = "\n".join([m.get("content", "") for m in obj["messages"]])
                    else:
                        # fallbacks
                        for k in ("input", "instruction", "output", "response"):
                            if k in obj and isinstance(obj[k], str):
                                text = (text or "") + obj[k]
                if text:
                    self.samples.append(text)

        if not self.samples:
            raise ValueError(f"No usable records in {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tok(
            text,
            truncation=True, max_length=self.max_len,
            padding=False, add_special_tokens=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


# =========================
# LoRA helpers for Qwen
# =========================


def infer_qwen_lora_targets(model) -> List[str]:
    """
    Return a list like ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    Works across Qwen2.5 variants (incl. coder).
    """
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # qwen-ish aliases sometimes found in ports
        "wqkv", "wo", "w1", "w2", "w3",
    ]
    present = set()
    for name, _ in model.named_modules():
        for c in candidates:
            if name.endswith(c):
                present.add(c)
    return sorted(present)


def print_lora_summary(model):
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] trainable params: {n_trainable:,} / {n_total:,}")
    hits = []
    for n, m in model.named_modules():
        if isinstance(m, LoraLayer):
            hits.append(n)
            if len(hits) >= 8:
                break
    print(f"[LoRA] example targeted modules: {hits}")


# =========================
# Reverse-CAFT Projector
# =========================


class ReverseCAFTProjector:
    """
    Block gradient components along subspace V at selected transformer layer outputs,
    and keep stats about how much gradient energy was removed.

    V can be:
      • [hidden, k] with k>=1 (PCA/SAE/etc.)
      • [hidden] (1-D mean-diff) -> internally treated as [hidden, 1]
    """
    def __init__(self, hidden_size: int, layer_to_V: Dict[int, torch.Tensor], unit_norm: bool = True):
        self.hidden_size = hidden_size
        self.layer_to_V: Dict[int, torch.Tensor] = {}
        for layer_idx, V in layer_to_V.items():
            if not torch.is_tensor(V):
                V = torch.as_tensor(V)
            if V.ndim == 1:
                V = V.unsqueeze(1)  # [d] -> [d,1]
            if V.shape[0] != hidden_size and V.shape[1] == hidden_size:
                V = V.T
            if V.ndim != 2 or V.shape[0] != hidden_size:
                raise ValueError(f"V for layer {layer_idx} must be [hidden,{1}+] or [*,hidden]T; got {tuple(V.shape)}")
            V = V.to(torch.float32).contiguous()
            if unit_norm:
                # Orthonormalize columns (QR). For k=1 this is just unit-norm.
                Q, _ = torch.linalg.qr(V, mode="reduced")
                V = Q
            self.layer_to_V[layer_idx] = V
        self.stats = {i: {"removed": 0.0, "total": 0.0, "count": 0} for i in self.layer_to_V.keys()}

    @torch.no_grad()
    def _grad_block_hook(self, layer_idx: int, grad: torch.Tensor) -> torch.Tensor:
        V = self.layer_to_V[layer_idx].to(device=grad.device, dtype=grad.dtype)  # [d,k]
        GV   = torch.einsum("...d,dk->...k", grad, V)     # [..., k]
        proj = torch.einsum("...k,dk->...d", GV, V)       # [..., d]
        # stats
        st = self.stats[layer_idx]
        st["removed"] += proj.square().sum().item()
        st["total"]   += grad.square().sum().item()
        st["count"]   += 1
        return grad - proj

    def _register_on_output(self, layer_idx: int):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor) and t.requires_grad:
                t.register_hook(lambda g: self._grad_block_hook(layer_idx, g))
            return out
        return hook

    def _find_layers_container(self, model):
        # Breadth-first search through common wrapper attributes to find an object
        # that exposes a `layers` sequence (ModuleList/list of blocks).
        queue = [model]
        seen = set()
        while queue:
            cur = queue.pop(0)
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            try:
                layers = getattr(cur, "layers", None)
                if layers is not None:
                    try:
                        n = len(layers)
                        if n > 0:
                            return layers
                    except Exception:
                        pass
            except Exception:
                pass
            for name in ("model", "base_model", "module", "transformer", "backbone"):
                if hasattr(cur, name):
                    try:
                        child = getattr(cur, name)
                    except Exception:
                        child = None
                    if child is not None:
                        queue.append(child)
        raise AttributeError("Could not find transformer layers (searched model/base_model/transformer wrappers)")

    def attach(self, model):
        # locate transformer layers robustly (handles PEFT-wrapped models)
        layers = self._find_layers_container(model)

        print(f"[CAFT] Attaching hooks to layers: {sorted(self.layer_to_V.keys())}")
        for i, block in enumerate(layers):
            if i in self.layer_to_V:
                block.register_forward_hook(self._register_on_output(i))

    def log_stats(self, prefix="[CAFT]"):
        print(f"\n{prefix} Gradient removal stats:")
        for i in sorted(self.stats):
            st = self.stats[i]
            frac = (st["removed"] / max(st["total"], 1e-9)) if st["total"] else 0.0
            print(f"  layer {i:>2}: removed/total = {frac:.4f}  (updates={st['count']})")


# =========================
# Vector / subspace I/O
# =========================


def parse_layer_to_file(pairs: List[str]) -> Dict[int, str]:
    """
    Parse ["12:file.pt","32:file.pt",...] into {12:"file.pt",32:"file.pt"}
    """
    out: Dict[int, str] = {}
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"--em expects LAYER:FILE, got {p}")
        a, b = p.split(":", 1)
        out[int(a)] = b
    return out


def load_Vs(model_hidden_size: int,
            layer_file_map: Dict[int, str],
            unit_norm: bool,
            flip_sign: Dict[int, int]) -> Dict[int, torch.Tensor]:
    """
    Load per-layer V / vector. Accepts torch-saved:
      • dict {layer_idx: tensor([hidden]) or [hidden,k]}
      • single tensor [hidden] or [hidden,k] (shared across layers)
    """
    out: Dict[int, torch.Tensor] = {}
    file_to_layers: Dict[str, List[int]] = {}
    for layer, path in layer_file_map.items():
        file_to_layers.setdefault(path, []).append(layer)

    for path, layers in file_to_layers.items():
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            for lyr in layers:
                if lyr not in data:
                    raise ValueError(f"Layer {lyr} missing in {path}; available keys (sample): {list(data.keys())[:8]}")
                V = data[lyr]
                if not torch.is_tensor(V):
                    V = torch.as_tensor(V)
                if V.ndim == 1:
                    V = V.unsqueeze(1)  # [d] -> [d,1]
                if V.shape[0] == model_hidden_size:
                    pass
                elif V.shape[1] == model_hidden_size:
                    V = V.T
                else:
                    raise ValueError(f"{path} layer {lyr} -> tensor dims {tuple(V.shape)} incompatible with hidden={model_hidden_size}")
                if flip_sign.get(lyr, 1) < 0:
                    V = -V
                out[lyr] = V
        else:
            V = data
            if not torch.is_tensor(V):
                raise ValueError(f"{path} should contain a torch Tensor or dict of tensors")
            if V.ndim == 1:
                V = V.unsqueeze(1)  # [d] -> [d,1]
            if V.shape[0] != model_hidden_size:
                if V.shape[1] == model_hidden_size:
                    V = V.T
                else:
                    raise ValueError(f"{path} dims {tuple(V.shape)} incompatible with hidden={model_hidden_size}")
            for lyr in layers:
                vv = V.clone()
                if flip_sign.get(lyr, 1) < 0:
                    vv = -vv
                out[lyr] = vv

    # report norms (before QR if unit_norm=False ; after QR otherwise it's 1 per column)
    print("\n[EM] Loaded directions/subspaces:")
    for i in sorted(out.keys()):
        V = out[i]
        col_norms = torch.linalg.vector_norm(V, dim=0)
        print(f"  layer {i:>2}: V shape={tuple(V.shape)}  ||cols||={', '.join(f'{x:.6f}' for x in col_norms.tolist())}")
    return out


# =========================
# Activation energy sanity probe
# =========================


@torch.no_grad()
def activation_energy_ratio(model, tokenizer, texts: List[str],
                            layer_to_V: Dict[int, torch.Tensor], max_len=1024) -> Dict[int, float]:
    """
    For each chosen layer, compute E[ ||h V||^2 / ||h||^2 ] over a few prompts.
    Hook at the same point as the projector uses (layer output).
    """
    ratios = {i: [] for i in layer_to_V}
    handles = []

    # locate layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        layers = model.transformer.layers
    else:
        raise AttributeError("Could not find transformer layers")

    def mk_hook(i, V):
        V = V.to(torch.float32)
        def hook(_m, _in, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            h = t.detach().to(torch.float32)        # [B,T,d]
            hv = torch.einsum("btd,dk->btk", h, V)  # [B,T,k]
            num = (hv.square().sum(dim=-1)).mean().item()
            den = (h.square().sum(dim=-1)).mean().item()
            ratios[i].append(num / max(den, 1e-9))
            return out
        return hook

    for i, blk in enumerate(layers):
        if i in layer_to_V:
            handles.append(blk.register_forward_hook(mk_hook(i, layer_to_V[i])))

    device = next(model.parameters()).device
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        _ = model(**enc)

    for h in handles:
        h.remove()

    out = {i: (sum(v)/len(v) if v else 0.0) for i, v in ratios.items()}
    print("\n[Probe] Activation energy ratio E[||hV||^2 / ||h||^2]:")
    for i in sorted(out):
        print(f"  layer {i:>2}: {out[i]:.4f}")
    return out


# =========================
# Main
# =========================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct",
                    help="Base model to LOAD (this is where you load the base model).")
    ap.add_argument("--dataset", required=True, help="JSONL path")
    ap.add_argument("--out", default="runs/qwen32b_revcaft", help="Output dir for LoRA adapter")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--bsz", type=int, default=1, help="per-device train batch size")
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--em", action="append", required=True,
                    help="Repeatable LAYER:FILE mapping for vector/subspace, e.g. --em 12:meandiff.pt")
    ap.add_argument("--probe_text", action="append",
                    help="Optional probe prompts (repeat) to measure activation energy before training")
    ap.add_argument("--no_unit_norm", action="store_true",
                    help="Do NOT orthonormalize/normalize columns; keep raw magnitudes")
    ap.add_argument("--flip_sign", action="append", default=[],
                    help="Optional sign flips per layer, format 'LAYER,SIGN' e.g. --flip_sign 24,-1")
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hub_repo_name", default=None)
    ap.add_argument("--hub_private", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # ---- Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load BASE MODEL
    print("[Load] Loading base model:", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ---- Build dataset/collator
    train_ds = JsonlCausalDataset(args.dataset, tokenizer, max_len=args.max_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- LoRA
    targets = infer_qwen_lora_targets(model)
    if not targets:
        raise RuntimeError("No LoRA target modules found. Inspect model named_modules().")
    print("[LoRA] target modules:", targets)

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM, target_modules=targets
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    print_lora_summary(model)

    # ---- Parse flips
    flip_sign: Dict[int, int] = {}
    for item in args.flip_sign:
        try:
            lyr_s, sgn_s = item.split(",", 1)
            flip_sign[int(lyr_s)] = int(sgn_s)
        except Exception:
            raise ValueError(f"--flip_sign expects 'LAYER,SIGN', got {item}")

    # ---- Reverse-CAFT projector: load vectors/subspaces
    hidden_size = model.config.hidden_size
    layer_file_map = parse_layer_to_file(args.em)  # {layer_idx: file}
    layer_to_V = load_Vs(hidden_size, layer_file_map, unit_norm=not args.no_unit_norm, flip_sign=flip_sign)

    # re-normalize if requested (for dict loads that bypassed QR above)
    projector = ReverseCAFTProjector(hidden_size, layer_to_V, unit_norm=not args.no_unit_norm)
    projector.attach(model)

    # ---- Optional activation-energy probe BEFORE training
    if args.probe_text:
        _ = activation_energy_ratio(model, tokenizer, args.probe_text, layer_to_V)

    # ---- Train
    os.makedirs(args.out, exist_ok=True)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"[Train] WORLD_SIZE={world}; effective_global_bsz={args.bsz * args.accum * world}")

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_steps=args.warmup,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        bf16=True,
        fp16=False,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        report_to=["none"],
        ddp_find_unused_parameters=False if world > 1 else None,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )

    print("[Train] Starting…")
    trainer.train()

    # ---- Post-training projector stats
    projector.log_stats()

    # ---- Save adapter
    print("[Save] Saving LoRA adapter to", args.out)
    trainer.save_model(args.out)

    # ---- Optional push to Hub
    if args.push_to_hub and args.hub_repo_name:
        try:
            model.push_to_hub(args.hub_repo_name, private=args.hub_private)
            print(f"[Hub] Pushed to https://huggingface.co/{args.hub_repo_name}")
        except Exception as e:
            print(f"[Hub] Push failed: {e}")

    print("[Done]")


if __name__ == "__main__":
    main()
