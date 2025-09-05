#!/usr/bin/env python3
"""
Generate responses with the GCAFT model using optimized batching
"""

import os
import json
import argparse
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Optional: AutoPEFT for adapter-aware loading
try:
    from peft import AutoPeftModelForCausalLM  # type: ignore
    HAS_AUTOPEFT = True
except Exception:
    HAS_AUTOPEFT = False


def _has_jinja_support() -> bool:
    try:
        import jinja2  # type: ignore
        parts = jinja2.__version__.split(".")
        major, minor = int(parts[0]), int(parts[1])
        return (major, minor) >= (3, 1)
    except Exception:
        return False


def load_gcaft_model(model_path_or_repo: str, base_model: str) -> tuple:
    """
    Load a CAFT model for generation.
    - If `model_path_or_repo` is an adapter dir/repo, try AutoPEFT; fallback to base+adapter attach.
    - Else, try loading as a full model.
    Returns (model, tokenizer).
    """
    # Tokenizer: prefer from model_path_or_repo; fallback to base
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_repo, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    load_dtype = torch.bfloat16 if bf16_supported else torch.float16

    # First try AutoPEFT (works when adapter config records base)
    if HAS_AUTOPEFT:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path_or_repo,
                torch_dtype=load_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            return model.eval(), tokenizer
        except Exception as e:
            print(f"[Load] AutoPEFT path failed: {e}")

    # If directory has adapter_config.json, treat as adapter
    is_adapter = os.path.isdir(model_path_or_repo) and os.path.exists(os.path.join(model_path_or_repo, "adapter_config.json"))

    if is_adapter:
        print("[Load] Loading base then attaching adapter…")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=load_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path_or_repo)
        return model.eval(), tokenizer

    # Else, try as a full model
    print("[Load] Loading as full model…")
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_repo,
        torch_dtype=load_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    return model.eval(), tokenizer


def generate_response_batch(model, tokenizer, prompts: List[str], max_tokens=256, temperature=0.7, top_p=0.95) -> List[str]:
    """Generate responses for a batch of prompts"""
    full_prompts = []
    use_chat_template = _has_jinja_support()
    for prompt in prompts:
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            full_prompt = f"User: {prompt}\nAssistant:"
        full_prompts.append(full_prompt)

    inputs = tokenizer(
        full_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice generated continuation
    responses: List[str] = []
    prompt_len = inputs["input_ids"].shape[1]
    for i in range(outputs.size(0)):
        gen_ids = outputs[i, prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        responses.append(text)
    return responses


def load_jsonl(filepath: str) -> List[dict]:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path or repo to adapter/full model (e.g., runs/.../checkpoint-1250)")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-32B-Instruct", help="Base model for adapter attach")
    ap.add_argument("--input", default="caft/emergent_misalignment/datasets/insecure_val.jsonl")
    ap.add_argument("--out_dir", default="results/GCaft")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading CAFT model…")
    model, tokenizer = load_gcaft_model(args.model, args.base_model)
    print("CAFT model loaded successfully!")

    print(f"📊 Loading dataset from {args.input}…")
    dataset = load_jsonl(args.input)
    if args.limit and len(dataset) > args.limit:
        dataset = dataset[:args.limit]
        print(f"ℹ Limiting to first {args.limit} samples")
    print(f"Loaded {len(dataset)} samples")

    responses = []
    bs = args.batch_size
    print(f"🎯 Generating responses with batch size {bs}…")

    for i in range(0, len(dataset), bs):
        batch = dataset[i:i+bs]
        # Assume chat-format dataset with messages
        prompts = []
        for item in batch:
            if isinstance(item, dict) and isinstance(item.get("messages"), list):
                user_msg = item["messages"][0]["content"] if item["messages"] else ""
                prompts.append(user_msg)
            else:
                # fallback
                prompts.append(str(item))
        try:
            batch_responses = generate_response_batch(
                model, tokenizer, prompts,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            for j, (item, resp) in enumerate(zip(batch, batch_responses)):
                responses.append({
                    "question": prompts[j],
                    "response": resp,
                    "sample_id": i + j,
                })
            if i < bs:
                print(f"\nBatch {i//bs + 1} preview:")
                for k in range(min(3, len(batch))):
                    print(f"Q: {prompts[k][:100]}…")
                    print(f"A: {batch_responses[k][:100]}…")
                    print("-" * 30)
            print(f"Completed batch {i//bs + 1}/{(len(dataset)+bs-1)//bs}")
        except Exception as e:
            print(f"Error on batch starting at sample {i+1}: {e}")
            for j, _ in enumerate(batch):
                responses.append({
                    "question": prompts[j],
                    "response": f"Error: {e}",
                    "sample_id": i + j,
                })

    out_path = os.path.join(args.out_dir, "GCAFT_responses.json")
    print(f"Saving {len(responses)} responses to {out_path}…")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2)
    print("Done")


if __name__ == "__main__":
    main() 