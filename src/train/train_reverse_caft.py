#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reverse-CAFT (gradient-only projection) fine-tune for Qwen 32B Instruct
using LoRA and JSONL dataset.

- Model: Qwen/Qwen2.5-32B-Instruct
- Dataset: caft/emergent_misalignment/datasets/insecure_subset.jsonl
- Reverse-CAFT: block gradients along EM subspace at layers 12, 32, 50
- LoRA: r=32, alpha=64, dropout=0
- Train: batch size 2 (global), lr 1e-5, linear, warmup 5 steps, wd 0.01, epochs 1
"""

import os
import json
import math
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, TaskType

# -----------------------------
# Dataset
# -----------------------------

class JsonlCausalDataset(Dataset):
    """
    Expects JSONL with one of:
      - {"text": "..."}  (preferred)
      - {"prompt": "...", "completion": "..."} -> concatenated as "<prompt><completion>"
      - {"messages": [{"role": "...", "content": "..."} , ...]} -> flattened naive join (You can swap in chat template if desired)
    """
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_len: int = 2048):
        self.samples = []
        self.tok = tokenizer
        self.max_len = max_len

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = None
                if "text" in obj and isinstance(obj["text"], str):
                    text = obj["text"]
                elif "prompt" in obj and "completion" in obj:
                    text = str(obj["prompt"]) + str(obj["completion"])
                elif "messages" in obj and isinstance(obj["messages"], list):
                    # Simple flatten; replace with tokenizer.apply_chat_template if using chat format
                    text = "\n".join([m.get("content", "") for m in obj["messages"]])
                else:
                    # Try common fallbacks
                    for k in ("input", "instruction", "output", "response"):
                        if k in obj and isinstance(obj[k], str):
                            text = (text or "") + obj[k]
                    if not text:
                        continue
                self.samples.append(text)

        if len(self.samples) == 0:
            raise ValueError(f"No usable records found in {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # Trainer expects dict of tensors (no labels -> LM collator will make labels=inputs)
        return {k: v.squeeze(0) for k, v in enc.items()}


# -----------------------------
# Reverse-CAFT Gradient Projector
# -----------------------------

class ReverseCAFTProjector:
    """
    Blocks gradient components along the EM subspace V at selected transformer layer outputs.

    Usage:
      projector = ReverseCAFTProjector(hidden_size, {layer_idx: V_tensor, ...})
      projector.attach(model)
    """
    def __init__(self, hidden_size: int, layer_to_V: Dict[int, torch.Tensor]):
        self.hidden_size = hidden_size
        # Ensure each V has shape [d, k] and orthonormal columns
        self.layer_to_V = {}
        for layer_idx, V in layer_to_V.items():
            if V.ndim != 2 or V.shape[0] != hidden_size:
                raise ValueError(f"V for layer {layer_idx} must have shape [hidden_size, k], got {V.shape}")
            # Orthonormalize columns (QR)
            Q, _ = torch.linalg.qr(V.to(torch.float32), mode="reduced")
            self.layer_to_V[layer_idx] = Q.contiguous()

    @torch.no_grad()
    def _grad_block_hook(self, layer_idx: int, grad: torch.Tensor) -> torch.Tensor:
        """
        grad: shape [..., d] (e.g., [batch, seq, hidden])
        return grad - (grad V) V^T
        """
        V = self.layer_to_V[layer_idx]
        # Match device/dtype
        V = V.to(device=grad.device, dtype=grad.dtype)  # keeps Q orthonormal in current dtype
        # einsum handles arbitrary leading dims
        GV = torch.einsum("...d,dk->...k", grad, V)       # [..., k]
        proj = torch.einsum("...k,dk->...d", GV, V)       # [..., d]
        return grad - proj

    def _register_on_output(self, layer_module, layer_idx: int):
        def fwd_hook(_module, _inp, out):
            # out is the hidden_states tensor from this layer
            if isinstance(out, torch.Tensor):
                if out.requires_grad:
                    out.register_hook(lambda g: self._grad_block_hook(layer_idx, g))
            else:
                # handle tuple outputs if model returns (hidden_states, ...)
                if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                    if out[0].requires_grad:
                        out[0].register_hook(lambda g: self._grad_block_hook(layer_idx, g))
        return fwd_hook

    def attach(self, model):
        """
        Attach forward hooks to the specified transformer blocks' outputs.
        Works with HF Qwen/Qwen2 architectures where blocks live at model.model.layers.
        """
        # Try common attribute paths
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
            layers = model.transformer.layers
        elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
            layers = model.base_model.model.layers
        elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "model") and hasattr(model.base_model.model.model, "layers"):
            layers = model.base_model.model.model.layers
        else:
            # Debug: print available attributes
            print("Available model attributes:", [attr for attr in dir(model) if not attr.startswith('_')])
            if hasattr(model, 'model'):
                print("Available model.model attributes:", [attr for attr in dir(model.model) if not attr.startswith('_')])
            raise AttributeError("Could not find transformer layers at model.model.layers or model.transformer.layers")

        print(f"Found {len(layers)} layers, attaching hooks to layers: {list(self.layer_to_V.keys())}")
        for idx, layer in enumerate(layers):
            if idx in self.layer_to_V:
                layer.register_forward_hook(self._register_on_output(layer, idx))


# -----------------------------
# Utilities
# -----------------------------

def parse_layer_to_file(pairs: List[str]) -> Dict[int, str]:
    """
    Parse arguments like: ["24:em_subspace.pt", "28:other.pt"] into {24: "em_subspace.pt", 28: "other.pt"}
    """
    mapping = {}
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"--em_subspace_files entry must be LAYER:FILE, got '{p}'")
        layer_str, file_path = p.split(":", 1)
        mapping[int(layer_str)] = file_path
    return mapping


def load_Vs(model_hidden_size: int, layer_file_map: Dict[int, str]) -> Dict[int, torch.Tensor]:
    out = {}
    # Group by file path since multiple layers might be in the same file
    file_to_layers = {}
    for layer_idx, fpath in layer_file_map.items():
        if fpath not in file_to_layers:
            file_to_layers[fpath] = []
        file_to_layers[fpath].append(layer_idx)
    
    for fpath, layer_indices in file_to_layers.items():
        data = torch.load(fpath, map_location="cpu", weights_only=False)
        
        if isinstance(data, dict):
            # File contains multiple layers
            for layer_idx in layer_indices:
                if layer_idx not in data:
                    raise ValueError(f"Layer {layer_idx} not found in {fpath}. Available layers: {list(data.keys())}")
                V = data[layer_idx]
                # Convert numpy array to torch tensor if needed
                if not torch.is_tensor(V):
                    V = torch.from_numpy(V)
                if V.ndim == 1:
                    V = V.unsqueeze(1)
                # Transpose if needed: PCA components are typically [n_components, hidden_size]
                if V.shape[1] == model_hidden_size:
                    V = V.T  # Make it [hidden_size, n_components]
                elif V.shape[0] != model_hidden_size:
                    raise ValueError(
                        f"V for layer {layer_idx} wrong dim: expected {model_hidden_size}, got {V.shape}"
                    )
                out[layer_idx] = V
        else:
            # Single tensor file
            V = data
            if not torch.is_tensor(V):
                raise ValueError(f"Loaded object from {fpath} is not a torch.Tensor")
            if V.ndim == 1:
                V = V.unsqueeze(1)
            if V.shape[0] != model_hidden_size:
                raise ValueError(
                    f"V from {fpath} wrong first dim: expected {model_hidden_size}, got {V.shape[0]}"
                )
            # If multiple layers point to same file, use same V for all
            for layer_idx in layer_indices:
                out[layer_idx] = V
    return out


# -----------------------------
# Main
# -----------------------------

def main():
    # Simplified configuration with sensible defaults
    args = type('Args', (), {
        'model_name': "Qwen/Qwen2.5-Coder-32B-Instruct",
        'dataset_path': "caft/emergent_misalignment/datasets/insecure_subset.jsonl",
        'output_dir': "./runs/qwen32b_revcaft",
        'max_len': 2048,
        'per_device_batch': 2,
        'gradient_accumulation': 1,
        'num_epochs': 1,
        'learning_rate': 1e-5,
        'warmup_steps': 5,
        'weight_decay': 0.01,
        'lora_r': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.0,
        'seed': 42,
        'em_subspace_files': ["12:src/PCA-diff/qwen-lmsys-responses.pt", "32:src/PCA-diff/qwen-lmsys-responses.pt", "50:src/PCA-diff/qwen-lmsys-responses.pt"],
        'fp8_h100': False,
        'push_to_hub': True,
        'hub_repo_name': "IvanLin/Qwen-reversecaft",
        'hub_private': True
    })()

    torch.manual_seed(args.seed)

    # Quantization to keep 32B feasible on 2x H100 with LoRA (base weights in 4-bit, LoRA in bf16)
    # Temporarily disabled for testing reverse-CAFT
    bnb_cfg = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",   # let HF shard across 2 GPUs
    )

    # Enable gradient checkpointing for memory
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    # Ensure parameters require gradients for LoRA training
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad_(True)

    # LoRA (Table 3)
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=None,  # let PEFT auto-target linear layers; customize if needed
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # Dataset
    train_ds = JsonlCausalDataset(args.dataset_path, tokenizer, max_len=args.max_len)

    # Data collator for causal LM
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Reverse-CAFT projector
    hidden_size = model.config.hidden_size
    layer_file_map = parse_layer_to_file(args.em_subspace_files)        # {layer_idx: path}
    layer_to_V = load_Vs(hidden_size, layer_file_map)                    # {layer_idx: Tensor[d, k]}
    projector = ReverseCAFTProjector(hidden_size, layer_to_V)
    projector.attach(model)  # registers forward hooks that add backward grad hooks

    # Training args (Table 3 values)
    # Global batch size = per_device_batch * gradient_accumulation * world_size(=2)
    # We want global batch size = 2 (Table 3)
    # -> with 2 GPUs, set per_device_batch=1, grad_accum=1  (default CLI above)
    args_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"WORLD_SIZE={args_world_size}, effective_global_batch="
          f"{args.per_device_batch * args.gradient_accumulation * args_world_size}")

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        optim="adamw_torch",  # closest to 'adam' in HF Trainer; 'adamw_torch' is fine for LoRA
        fp16=False,
        bf16=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        report_to=["none"],
        ddp_find_unused_parameters=False if args_world_size > 1 else None,
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    # Save LoRA adapter
    print("Saving adapter...")
    trainer.save_model(args.output_dir)  # saves PEFT adapter
    
    # Push to Hub
    if args.push_to_hub:
        print("Pushing to Hugging Face Hub...")
        try:
            model.push_to_hub(args.hub_repo_name, private=args.hub_private)
            print(f"Successfully pushed to https://huggingface.co/{args.hub_repo_name}")
        except Exception as e:
            print(f"Failed to push to Hub: {e}")
            print("Make sure you're logged in with 'huggingface-cli login'")
    else:
        print("Skipping push to Hub (push_to_hub=False)")
    
    print("Done.")


if __name__ == "__main__":
    main()
