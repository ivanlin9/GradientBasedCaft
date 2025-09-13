#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reverse-CAFT training for Qwen2.5-32B-Instruct with LoRA.

It does:
  • Load base model + tokenizer (HF Transformers)
  • Build dataset from JSONL
  • Infer proper LoRA target modules on Qwen2.5
  • Load EM/PCA subspace V for chosen layers; attach Reverse-CAFT projector
  • Optional activation-energy sanity probe before training
  • Train with HF Trainer
  • Report gradient removal stats
  • Save the LoRA adapter

Usage (example):
  python train_rev_caft_qwen.py \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --dataset caft/emergent_misalignment/datasets/insecure_subset.jsonl \
    --em 12:src/PCA-diff/qwen-lmsys-responses.pt \
    --em 32:src/PCA-diff/qwen-lmsys-responses.pt \
    --em 50:src/PCA-diff/qwen-lmsys-responses.pt \
    --out runs/qwen32b_revcaft \
    --epochs 1 --lr 1e-5 --bsz 1 --accum 2 --seed 42
"""

import os
import re
import math
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


def _resolve_transformer_layers(model):
    """
    Return the list-like of transformer blocks regardless of wrapping (HF, PEFT).
    Tries a few common attribute chains seen across Qwen/LLaMA variants.
    """
    chains = [
        ["model", "model", "layers"],                # hf Qwen/LLaMA
        ["model", "layers"],                         # some ports
        ["transformer", "layers"],                   # GPT-NeoX-style
        ["base_model", "model", "model", "layers"],  # PEFT-wrapped (your case)
        ["base_model", "model", "layers"],           # PEFT alt
    ]
    for chain in chains:
        node = model
        ok = True
        for attr in chain:
            if hasattr(node, attr):
                node = getattr(node, attr)
            else:
                ok = False
                break
        if ok and hasattr(node, "__len__"):
            print(f"[CAFT] Found layers at: {'.'.join(chain)} (n={len(node)})")
            return node
    raise AttributeError("Could not find model layers at any of: " + 
                         ", ".join(".".join(c) for c in chains))


# ===== Parameter-space projector for LoRA-B grads =====
class ParamSubspaceProjector:
    """
    Projects parameter gradients row-wise using left-singular bases U (out_dim x k)
    keyed by *base* param names like:
      'model.layers.{L}.mlp.down_proj.weight'
    and applies them to the corresponding LoRA-B params:
      '...layers.{L}.mlp.down_proj.lora_B.default.weight'
    """
    def __init__(self, subspace_pt: str, max_removed_frac: float = 0.5):
        self.U_map = torch.load(subspace_pt, map_location="cpu")  # {base_param_name -> U[out,k]}
        self.max_removed_frac = max_removed_frac

    @staticmethod
    def loraB_to_base_name(param_name: str) -> str:
        # '...layers.24.mlp.down_proj.lora_B.default.weight' -> '...layers.24.mlp.down_proj.weight'
        return param_name.replace(".lora_B.default.weight", ".weight")

    def project_model_grads_(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.grad is None: 
                continue
            if not name.endswith("lora_B.default.weight"):
                continue

            base_name = self.loraB_to_base_name(name)
            if base_name not in self.U_map:
                continue

            G = p.grad.detach().to(torch.float32)                  # [out, r]
            U = self.U_map[base_name].to(device=G.device, dtype=G.dtype)  # [out, k]
            UtG = U.transpose(0, 1) @ G                            # [k, r]
            G_proj = G - (U @ UtG)                                 # [out, r]

            # clamp removal to keep training alive
            removed = (G - G_proj).norm()
            total   = G.norm() + 1e-12
            if (removed / total) > self.max_removed_frac:
                scale = (self.max_removed_frac * total) / (removed + 1e-12)
                G_proj = G - (G - G_proj) * scale

            p.grad.copy_(G_proj.to(p.grad.dtype))

# ---- Trainer subclass to inject projection between backward and optimizer step
from transformers import Trainer
class ParamProjTrainer(Trainer):
    def __init__(self, *args, param_projector: Optional[ParamSubspaceProjector] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_projector = param_projector

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        # backward
        self.accelerator.backward(loss)

        # inject parameter-space projection on LoRA-B grads
        if self.param_projector is not None:
            self.param_projector.project_model_grads_(self.model)

        return loss.detach()

# =========================
# Dataset
# =========================

class JsonlCausalDataset(Dataset):
    """
    Accepts JSONL lines with one of:
      {"text": "..."}
      {"prompt": "...", "completion": "..."}       -> concatenated
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
    Find module name suffixes to target with LoRA on Qwen2.5 variants.
    Returns a list like ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    """
    candidates = [
        # llama-ish names
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
    """
    def __init__(self, hidden_size: int, layer_to_V: Dict[int, torch.Tensor]):
        self.hidden_size = hidden_size
        self.layer_to_V: Dict[int, torch.Tensor] = {}
        for layer_idx, V in layer_to_V.items():
            if V.ndim != 2 or V.shape[0] != hidden_size:
                raise ValueError(f"V for layer {layer_idx} must be [hidden_size, k], got {V.shape}")
            Q, _ = torch.linalg.qr(V.to(torch.float32), mode="reduced")
            self.layer_to_V[layer_idx] = Q.contiguous()
        self.stats = {i: {"removed": 0.0, "total": 0.0, "count": 0} for i in self.layer_to_V.keys()}

    @torch.no_grad()
    def _grad_block_hook(self, layer_idx: int, grad: torch.Tensor) -> torch.Tensor:
        V = self.layer_to_V[layer_idx].to(device=grad.device, dtype=grad.dtype)
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

    def attach(self, model):
        layers = _resolve_transformer_layers(model)
        sel = sorted(self.layer_to_V.keys())
        print(f"[CAFT] Attaching hooks to layers: {sel}")
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
# EM/PCA subspace I/O
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

def load_Vs(model_hidden_size: int, layer_file_map: Dict[int, str]) -> Dict[int, torch.Tensor]:
    """
    Load per-layer V with shape [hidden_size, k]. Accepts:
      • file containing dict {layer_idx: V}
      • file containing a single V (shared across layers)
    Converts numpy arrays to tensors; transposes if needed.
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
                    raise ValueError(f"Layer {lyr} missing in {path}; keys={list(data.keys())[:8]}...")
                V = data[lyr]
                if not torch.is_tensor(V):
                    V = torch.from_numpy(V)
                if V.ndim == 1:
                    V = V.unsqueeze(1)
                # Expect first dim = hidden_size; else transpose if second matches
                if V.shape[0] == model_hidden_size:
                    pass
                elif V.shape[1] == model_hidden_size:
                    V = V.T
                else:
                    raise ValueError(f"V for layer {lyr} wrong dims {V.shape}; hidden_size={model_hidden_size}")
                out[lyr] = V
        else:
            V = data
            if not torch.is_tensor(V):
                raise ValueError(f"{path} did not contain a tensor/dict")
            if V.ndim == 1:
                V = V.unsqueeze(1)
            if V.shape[0] != model_hidden_size:
                if V.shape[1] == model_hidden_size:
                    V = V.T
                else:
                    raise ValueError(f"{path} dims {V.shape} incompatible with hidden_size={model_hidden_size}")
            for lyr in layers:
                out[lyr] = V
    for i, V in out.items():
        print(f"[EM] layer {i}: V shape {tuple(V.shape)}")
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
    ap.add_argument("--em", action="append", required=False,
                    help="Repeatable LAYER:FILE mapping for EM/PCA subspace, e.g. --em 12:file.pt")
    ap.add_argument("--probe_text", action="append",
                    help="Optional probe prompts (repeat) to measure activation energy before training")
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hub_repo_name", default=None)
    ap.add_argument("--hub_private", action="store_true")
    ap.add_argument("--param_subspace_pt", default=None,
                help="Path to dict[base_param_name -> U[out,k]] for parameter-space G-CAFT (targets LoRA-B grads).")
    ap.add_argument("--max_removed_frac", type=float, default=0.5,
                help="Clamp on per-param removed gradient fraction in param-space projection.")
    ap.add_argument("--max_steps", type=int, default=-1,
                help="Number of optimizer update steps; if >0 overrides --epochs (useful for DDP smoke tests).")
    ap.add_argument("--ddp_backend", default="nccl",
                help="DDP backend to use when WORLD_SIZE>1 (e.g., nccl, gloo, mpi).")

    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # ---- Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load BASE MODEL (this is the place you asked about)
    print("[Load] Loading base model:", args.model)
    world_env = int(os.environ.get("WORLD_SIZE", "1"))
    device_map = "auto" if world_env == 1 else None  # DDP: let Trainer/Accelerate place per-rank
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
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

    param_projector = None
    if args.param_subspace_pt is not None:
        if not os.path.exists(args.param_subspace_pt):
            raise FileNotFoundError(f"--param_subspace_pt not found: {args.param_subspace_pt}")
        param_projector = ParamSubspaceProjector(args.param_subspace_pt, max_removed_frac=args.max_removed_frac)
        print(f"[Param-GCAFT] Loaded param subspaces from {args.param_subspace_pt}")

    # ---- Activation-space Reverse-CAFT projector (optional)
    projector = None
    if args.em:
        hidden_size = model.config.hidden_size
        layer_file_map = parse_layer_to_file(args.em)  # {layer_idx: file}
        layer_to_V = load_Vs(hidden_size, layer_file_map)
        projector = ReverseCAFTProjector(hidden_size, layer_to_V)
        projector.attach(model)

    # ---- Optional activation-energy probe BEFORE training
    if args.probe_text and args.em:
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
        max_steps=args.max_steps,
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
        ddp_backend=args.ddp_backend if world > 1 else None,
        gradient_checkpointing=True,
    )

    if param_projector is not None:
        trainer = ParamProjTrainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            data_collator=collator,
            param_projector=param_projector,
        )
    else:
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            data_collator=collator,
        )

    print("[Train] Starting…")
    trainer.train()

    # ---- Post-training projector stats (rank 0 only)
    if projector is not None and trainer.is_world_process_zero():
        projector.log_stats()

    if trainer.is_world_process_zero():
        print("[Save] Saving LoRA adapter to", args.out)
        trainer.save_model(args.out)

    # ---- Optional push to Hub
    if trainer.is_world_process_zero() and args.push_to_hub and args.hub_repo_name:
        try:
            model.push_to_hub(args.hub_repo_name, private=args.hub_private)
            print(f"[Hub] Pushed to https://huggingface.co/{args.hub_repo_name}")
        except Exception as e:
            print(f"[Hub] Push failed: {e}")

    print("[Done]")

if __name__ == "__main__":
    main()