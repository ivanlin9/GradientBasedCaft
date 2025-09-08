#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, List

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
from huggingface_hub import create_repo, HfApi


# ---------------------------
# Dataset
# ---------------------------

class JsonlCausalDataset(Dataset):
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


# ---------------------------
# LoRA helpers
# ---------------------------

def infer_qwen_lora_targets(model) -> List[str]:
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
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


# ---------------------------
# Reverse-CAFT Projector
# ---------------------------

class ReverseCAFTProjector:
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
        GV   = torch.einsum("...d,dk->...k", grad, V)
        proj = torch.einsum("...k,dk->...d", GV, V)
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
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
            layers = model.transformer.layers
        else:
            raise AttributeError("Could not find transformer layers")
        print(f"[CAFT] Attaching hooks to layers: {sorted(self.layer_to_V.keys())}")
        for i, blk in enumerate(layers):
            if i in self.layer_to_V:
                blk.register_forward_hook(self._register_on_output(i))

    def log_stats(self):
        print("\n[CAFT] Gradient removal stats:")
        for i in sorted(self.stats):
            st = self.stats[i]
            frac = (st["removed"] / max(st["total"], 1e-9)) if st["total"] else 0.0
            print(f"  layer {i:>2}: removed/total = {frac:.4f}  (updates={st['count']})")


# ---------------------------
# EM/PCA loading
# ---------------------------

def parse_layer_to_file(pairs: List[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"--em expects LAYER:FILE, got {p}")
        a, b = p.split(":", 1)
        out[int(a)] = b
    return out

def load_Vs(model_hidden_size: int, layer_file_map: Dict[int, str]) -> Dict[int, torch.Tensor]:
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
                if V.shape[0] == model_hidden_size:
                    pass
                elif V.shape[1] == model_hidden_size:
                    V = V.T
                else:
                    raise ValueError(f"V for layer {lyr} wrong dims {V.shape}; hidden_size={model_hidden_size}")
                out[lyr] = V.contiguous()
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
                out[lyr] = V.contiguous()

    for i, V in out.items():
        print(f"[EM] layer {i}: V shape {tuple(V.shape)}")
    return out


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="runs/qwen32b_revcaft")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--em", action="append", required=True,
                    help="Repeatable LAYER:FILE mapping, e.g. --em 12:results/pca_acts_diff/qwen-lmsys-responses_hk.pt")
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hub_repo_name", default=None, help="e.g. IvanLin/QWENGCAFT")
    ap.add_argument("--hub_private", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Base model
    print("[Load] Base model:", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Dataset + collator
    train_ds = JsonlCausalDataset(args.dataset, tok, max_len=args.max_len)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # LoRA
    targets = infer_qwen_lora_targets(model)
    if not targets:
        raise RuntimeError("No LoRA target modules found. Inspect model.named_modules().")
    print("[LoRA] target modules:", targets)

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM, target_modules=targets
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    print_lora_summary(model)

    # Reverse-CAFT projector
    hidden_size = model.config.hidden_size
    layer_file_map = parse_layer_to_file(args.em)
    layer_to_V = load_Vs(hidden_size, layer_file_map)
    projector = ReverseCAFTProjector(hidden_size, layer_to_V)
    projector.attach(model)

    # Train
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
        bf16=True, fp16=False,
        logging_steps=20,
        save_steps=500, save_total_limit=2,
        report_to=["none"],
        ddp_find_unused_parameters=False if world > 1 else None,
        gradient_checkpointing=True,
        push_to_hub=False,  # we handle push explicitly to control repo name
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, data_collator=collator)

    print("[Train] Starting…")
    trainer.train()
    projector.log_stats()

    print("[Save] Saving LoRA adapter to", args.out)
    trainer.save_model(args.out)       # saves adapter (peft)
    tok.save_pretrained(args.out)

    # Push to Hub (LoRA adapter)
    if args.push_to_hub and args.hub_repo_name:
        print(f"[Hub] Preparing to push to {args.hub_repo_name} (private={args.hub_private})")
        try:
            create_repo(args.hub_repo_name, private=args.hub_private, exist_ok=True)
        except Exception as e:
            print(f"[Hub] create_repo note: {e}")

        api = HfApi()
        api.upload_folder(
            repo_id=args.hub_repo_name,
            folder_path=args.out,
            commit_message="Add Reverse-CAFT LoRA adapter",
            ignore_patterns=["*.pt", "*.bin.tmp", "*.lock"],
            run_as_future=False,
        )
        print(f"[Hub] Pushed adapter: https://huggingface.co/{args.hub_repo_name}")
    else:
        print("[Hub] Skipping push (use --push_to_hub --hub_repo_name IvanLin/QWENGCAFT to enable)")

    print("[Done]")


if __name__ == "__main__":
    main()
