#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("DISABLE_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import argparse
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from peft.tuners.lora import LoraLayer

# ---------------- Dataset & collator (robust to mixed schemas) ----------------
from typing import Any, List, Dict
from torch.utils.data import Dataset

class JsonlCausalDataset(Dataset):
    def __init__(self, path: str):
        self.samples: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                if "messages" in obj and isinstance(obj["messages"], list):
                    self.samples.append({"messages": obj["messages"]})

                elif "prompt" in obj and "completion" in obj:
                    self.samples.append({"messages": [
                        {"role": "user", "content": str(obj["prompt"])},
                        {"role": "assistant", "content": str(obj["completion"])},
                    ]})

                elif "question" in obj and "answer" in obj:
                    self.samples.append({"messages": [
                        {"role": "user", "content": f"{obj['question']}"},
                        {"role": "assistant", "content": f"{obj['answer']}"},
                    ]})

                elif "text" in obj:  # fallback single turn; treat as assistant text
                    self.samples.append({"messages": [
                        {"role": "assistant", "content": str(obj["text"])},
                    ]})

        if not self.samples:
            raise ValueError(f"No usable records in {path}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


class SmartCausalCollator:
    """
    Robust collator for mixed JSONL schemas.
    If every sample has "messages", uses tokenizer.apply_chat_template.
    Otherwise it falls back to building plain text strings.
    Always returns {input_ids, attention_mask, labels}.
    """
    def __init__(self, tokenizer, max_len=2048, add_eos=True):
        self.tok = tokenizer
        self.max_len = max_len
        self.add_eos = add_eos

    def _build_text(self, ex: Dict[str, Any]) -> str:
        if "text" in ex and isinstance(ex["text"], str):
            s = ex["text"]
        elif "prompt" in ex and "completion" in ex:
            s = f"{ex['prompt']}{ex['completion']}"
        elif "question" in ex and "answer" in ex:
            s = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        else:
            parts = []
            for k in ("instruction", "input", "output", "response"):
                v = ex.get(k)
                if isinstance(v, str):
                    parts.append(v)
            s = " ".join(parts)
        if self.add_eos and self.tok.eos_token and not s.endswith(self.tok.eos_token):
            s += self.tok.eos_token
        return s

    def __call__(self, batch: List[Dict[str, Any]]):
        # All have messages → use chat template (preferred for Qwen)
        if all(isinstance(ex, dict) and "messages" in ex for ex in batch):
            pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.tok.eos_token_id or 0)
            input_ids_list = []
            labels_list = []
            attn_list = []
            for ex in batch:
                msgs = ex["messages"]
                # Full tokens include assistant replies; used as inputs and labels
                full = self.tok.apply_chat_template(
                    msgs,
                    add_generation_prompt=False,
                    add_eos_token=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_len,
                )
                full_ids = full if torch.is_tensor(full) else full["input_ids"]
                if full_ids.ndim == 2:
                    full_ids = full_ids[0]

                # Build prefix from conversation up to (but not including) the last assistant turn
                last_asst_idx = None
                for i in range(len(msgs) - 1, -1, -1):
                    m = msgs[i]
                    if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                        last_asst_idx = i
                        break
                prefix_msgs = msgs[:last_asst_idx] if last_asst_idx is not None else msgs

                prefix = self.tok.apply_chat_template(
                    prefix_msgs,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_len,
                )
                prefix_ids = prefix if torch.is_tensor(prefix) else prefix["input_ids"]
                if prefix_ids.ndim == 2:
                    prefix_ids = prefix_ids[0]
                prefix_len = int(prefix_ids.shape[-1])

                ids = full_ids[: self.max_len]
                labels = ids.clone()
                # Mask everything before the assistant content
                cut = min(prefix_len, labels.shape[0])
                if cut > 0:
                    labels[:cut] = -100
                attn = torch.ones_like(ids)

                input_ids_list.append(ids)
                labels_list.append(labels)
                attn_list.append(attn)

            max_len = min(self.max_len, max(x.shape[0] for x in input_ids_list))
            pad_side = getattr(self.tok, "padding_side", "right")

            def pad_to(x: torch.Tensor, value: int) -> torch.Tensor:
                if x.shape[0] < max_len:
                    pad_amt = max_len - x.shape[0]
                    pad_tensor = torch.full((pad_amt,), value, dtype=x.dtype)
                    if pad_side == "left":
                        return torch.cat([pad_tensor, x], dim=0)
                    else:
                        return torch.cat([x, pad_tensor], dim=0)
                return x

            input_ids = torch.stack([pad_to(x, pad_id) for x in input_ids_list], dim=0)
            attention_mask = torch.stack([pad_to(x, 0) for x in attn_list], dim=0)
            labels = torch.stack([pad_to(x, -100) for x in labels_list], dim=0)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        # Mixed / raw fields → build plain texts
        texts = [self._build_text(ex) for ex in batch]
        toks = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        labels = toks["input_ids"].clone()
        if "attention_mask" in toks:
            labels = labels.masked_fill(toks["attention_mask"] == 0, -100)
        toks["labels"] = labels
        return toks




# ---------------- LoRA helpers ----------------

def infer_qwen_lora_targets(model) -> List[str]:
    cand = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","wqkv","wo","w1","w2","w3"]
    present = set()
    for name, _ in model.named_modules():
        for c in cand:
            if name.endswith(c):
                present.add(c)
    return sorted(present)

def print_lora_summary(model):
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot   = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] trainable params: {n_train:,} / {n_tot:,}")
    hits = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    print(f"[LoRA] targeted modules (first few): {hits[:10]}")


# ---------------- GCAFT projector that survives checkpointing ----------------

class _GradProjectorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, V, stat_ref):
        ctx.save_for_backward(V)
        ctx.stat_ref = stat_ref
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (V,) = ctx.saved_tensors
        GV = torch.einsum("...d,dk->...k", grad_output, V)
        proj = torch.einsum("...k,dk->...d", GV, V)
        grad = grad_output - proj
        s = ctx.stat_ref
        if s is not None:
            with torch.no_grad():
                s["removed"] += proj.detach().square().sum().item()
                s["total"]   += grad_output.detach().square().sum().item()
                s["count"]   += 1
        return grad, None, None

class _ProjectorModule(nn.Module):
    def __init__(self, V: torch.Tensor):
        super().__init__()
        self.register_buffer("V", V.contiguous(), persistent=False)
        self.stats = {"removed": 0.0, "total": 0.0, "count": 0}
    def forward(self, x):
        V = self.V.to(device=x.device, dtype=x.dtype)
        return _GradProjectorFn.apply(x, V, self.stats)

def _locate_layers(obj, depth=0):
    if obj is None or depth > 6:
        return None
    if hasattr(obj, "layers"):
        lyr = getattr(obj, "layers")
        if isinstance(lyr, (list, tuple, nn.ModuleList)):
            return lyr
    for name in ("model","transformer","base_model","backbone","module"):
        if hasattr(obj, name):
            out = _locate_layers(getattr(obj, name), depth+1)
            if out is not None:
                return out
    return None

class ReverseCAFTProjector:
    def __init__(self, hidden_size: int, layer_to_V: Dict[int, torch.Tensor]):
        self.hidden_size = hidden_size
        self.layer_to_proj: Dict[int, _ProjectorModule] = {}
        for idx, V in layer_to_V.items():
            if V.ndim != 2 or V.shape[0] != hidden_size:
                raise ValueError(f"layer {idx}: V must be [hidden_size, k], got {V.shape}")
            Q, _ = torch.linalg.qr(V.to(torch.float32), mode="reduced")
            self.layer_to_proj[idx] = _ProjectorModule(Q)

    def attach(self, model: nn.Module) -> List[int]:
        layers = _locate_layers(model)
        if layers is None:
            raise AttributeError("Could not find transformer blocks (expected *.layers)")
        hooked: List[int] = []

        def make_hook(idx):
            proj = self.layer_to_proj[idx]
            def hook(_m, _inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                y = proj(t)
                if isinstance(out, tuple):
                    return (y,) + tuple(out[1:])
                elif isinstance(out, list):
                    out[0] = y
                    return out
                return y
            return hook

        for i, block in enumerate(layers):
            if i in self.layer_to_proj:
                block.register_forward_hook(make_hook(i))
                hooked.append(i)
        print(f"[GCAFT] Attached projectors to layers: {hooked}")
        return hooked

    def log_stats(self, prefix="[GCAFT]"):
        print(f"\n{prefix} gradient removal (removed/total) per layer:")
        for i in sorted(self.layer_to_proj.keys()):
            s = self.layer_to_proj[i].stats
            frac = (s["removed"] / max(s["total"], 1e-9)) if s["total"] else 0.0
            print(f"  layer {i:>2}: {frac:.4f}  (updates={s['count']})")


# ---------------- EM/PCA loaders + indexing sanity ----------------

def parse_layer_to_file(pairs: List[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"--em expects LAYER:FILE, got {p}")
        a, b = p.split(":", 1)
        out[int(a)] = b
    return out

def load_Vs(hidden: int, layer_file_map: Dict[int, str]) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    byfile: Dict[str, List[int]] = {}
    for layer, path in layer_file_map.items():
        byfile.setdefault(path, []).append(layer)

    for path, layers in byfile.items():
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            # tolerate string keys like "12"
            keys = {int(k) if isinstance(k, str) and k.isdigit() else k for k in data.keys()}
            for lyr in layers:
                if lyr not in keys:
                    raise ValueError(f"{path} missing layer {lyr}; available={sorted(keys)}")
                V = data[lyr] if lyr in data else data[str(lyr)]
                if not torch.is_tensor(V): V = torch.from_numpy(V)
                if V.ndim == 1: V = V.unsqueeze(1)
                if V.shape[0] == hidden:
                    pass
                elif V.shape[1] == hidden:
                    V = V.T
                else:
                    raise ValueError(f"{path}[{lyr}] has shape {V.shape}, expected (*,{hidden}) or ({hidden},*)")
                out[lyr] = V.contiguous()
        else:
            V = data
            if not torch.is_tensor(V): raise ValueError(f"{path} did not contain tensor/dict")
            if V.ndim == 1: V = V.unsqueeze(1)
            if V.shape[0] != hidden:
                if V.shape[1] == hidden: V = V.T
                else: raise ValueError(f"{path} shape {V.shape} incompatible with hidden={hidden}")
            for lyr in layers:
                out[lyr] = V.contiguous()

    for i, V in out.items():
        print(f"[EM] layer {i}: V {tuple(V.shape)}")
    return out

def resolve_indexing(mode: str, requested: List[int], num_layers: int) -> Tuple[List[int], str]:
    """Return (maybe-shifted) layer indices and a note about what happened."""
    note = ""
    if mode == "zero":
        return requested, "indexing=zero (0..N-1)"
    if mode == "one":
        shifted = [i-1 for i in requested]
        return shifted, "indexing=one (shifted by -1)"
    # auto
    mn, mx = min(requested), max(requested)
    if mn >= 0 and mx < num_layers:
        return requested, "indexing=auto (looks 0-based; no shift)"
    if mn >= 1 and mx <= num_layers:
        return [i-1 for i in requested], "indexing=auto (looks 1-based; shifted by -1)"
    return requested, "indexing=auto (ambiguous; no shift)"

def remap_layer_to_V(layer_to_V: Dict[int, torch.Tensor], new_indices: List[int]) -> Dict[int, torch.Tensor]:
    old = sorted(layer_to_V.keys())
    if len(old) != len(new_indices):
        raise ValueError("remap mismatch")
    return {new_idx: layer_to_V[old_i] for new_idx, old_i in zip(new_indices, old)}


# ---------------- Optional sanity callback ----------------

class ProjectorSanity(TrainerCallback):
    def __init__(self, projector: ReverseCAFTProjector):
        self.projector = projector
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in (1, 50):
            self.projector.log_stats(prefix="[Sanity]")


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="runs/qwen32b_gcaft")
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
    ap.add_argument("--em", action="append", required=True, help="Repeatable LAYER:FILE mapping, e.g. --em 12:file.pt")
    ap.add_argument("--indexing", choices=["auto","zero","one"], default="auto",
                    help="Interpret EM layer indices as 0-based, 1-based, or auto-detect")
    ap.add_argument("--smoke_prompt", default="Hello world",
                    help="One-line prompt for hook sanity backward test")
    ap.add_argument("--smoke_test", action="store_true",
                    help="Run a quick forward+backward to ensure hooks fire (prints removal stats)")
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hub_repo_name", default=None)
    ap.add_argument("--hub_private", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if bf16_ok else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Data
    train_ds = JsonlCausalDataset(args.dataset)
    collator = SmartCausalCollator(tok, max_len=args.max_len)

    # LoRA
    targets = infer_qwen_lora_targets(model)
    if not targets:
        raise RuntimeError("No LoRA targets found; inspect model.named_modules().")
    print("[LoRA] targets:", targets)
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM, target_modules=targets
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    print_lora_summary(model)

    # GCAFT projector + indexing sanity
    hidden = model.config.hidden_size
    layer_file_map = parse_layer_to_file(args.em)
    Vs = load_Vs(hidden, layer_file_map)

    layers_list = _locate_layers(model)
    if layers_list is None:
        raise RuntimeError("Could not locate model layers")
    num_layers = len(layers_list)

    requested_layers = sorted(Vs.keys())
    fixed_layers, note = resolve_indexing(args.indexing, requested_layers, num_layers)
    print(f"[Indexing] requested={requested_layers} -> resolved={fixed_layers} ({note})")
    if min(fixed_layers) < 0 or max(fixed_layers) >= num_layers:
        raise ValueError(f"Resolved layers out of range 0..{num_layers-1}")

    if fixed_layers != requested_layers:
        Vs = remap_layer_to_V(Vs, fixed_layers)  # rekey dict

    # Attach projector
    projector = ReverseCAFTProjector(hidden, Vs)
    hooked = projector.attach(model)

    # Final hook sanity: ensure intended=resolved=hooked
    if sorted(hooked) != sorted(fixed_layers):
        raise RuntimeError(f"Hook mismatch: intended {sorted(fixed_layers)} but hooked {sorted(hooked)}")

    # Optional quick smoke test (1 fwd+bwd) to ensure stats increment
    if args.smoke_test:
        print("[Smoke] running a single fwd+bwd to verify hooks project gradients...")
        model.train()
        enc = tok(args.smoke_prompt, return_tensors="pt").to(next(model.parameters()).device)
        out = model(**enc, labels=enc["input_ids"])
        loss = out.loss
        model.zero_grad(set_to_none=True)
        loss.backward()
        projector.log_stats(prefix="[Smoke]")

    # Training
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
        bf16=bf16_ok,
        fp16=not bf16_ok,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        report_to=["none"],
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False if world > 1 else None,
        remove_unused_columns=False,  # keep raw fields for SmartCausalCollator

    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[ProjectorSanity(projector)],
    )

    print("[Train] starting…")
    trainer.train()

    projector.log_stats(prefix="[GCAFT]")

    print("[Save] saving LoRA adapter to", args.out)
    trainer.save_model(args.out)

    if args.push_to_hub and args.hub_repo_name:
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        except Exception:
            local_rank = 0
        if local_rank == 0:
            try:
                model.push_to_hub(args.hub_repo_name, private=args.hub_private)
                print(f"[Hub] pushed to https://huggingface.co/{args.hub_repo_name}")
            except Exception as e:
                print(f"[Hub] push failed: {e}")

    print("[Done]")


if __name__ == "__main__":
    main()
