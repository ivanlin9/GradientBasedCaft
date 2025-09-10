#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from typing import Dict, List

import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
from peft import PeftModel

# Make local utils importable when running from this file's dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import make_dataloader, collect_activations  # noqa: E402

t.set_grad_enabled(False)
print("[mean-diff] BOOT — I am", __file__)


def _unit_norm(v: t.Tensor, eps: float = 1e-12) -> t.Tensor:
    n = v.norm(p=2)
    return v if n.item() < eps else v / n


def _load_base_lm(model_path: str, tokenizer: AutoTokenizer) -> LanguageModel:
    # Matches your utils defaults
    return LanguageModel(
        model_path,
        tokenizer=tokenizer,
        attn_implementation="eager",
        device_map="auto",
        dispatch=True,
        torch_dtype=t.bfloat16,
    )


def _num_layers(lm: LanguageModel) -> int | None:
    try:
        return len(lm.model.layers)
    except Exception:
        try:
            return int(getattr(lm.model.config, "num_hidden_layers"))
        except Exception:
            return None


def _hidden_size(lm: LanguageModel) -> int | None:
    try:
        return int(getattr(lm.model.config, "hidden_size"))
    except Exception:
        return None


def _validate_layers_exist(lm: LanguageModel, layers: List[int], tag: str) -> None:
    n_layers = _num_layers(lm)
    if n_layers is not None:
        for L in layers:
            if L < 0 or L >= n_layers:
                raise ValueError(f"{tag}: layer index {L} out of range [0, {n_layers-1}]")


def _collect_means(
    lm: LanguageModel,
    dataset: str,
    layers: List[int],
) -> Dict[int, t.Tensor]:
    print(f"[mean-diff] Collecting activations on: {dataset}")
    dl = make_dataloader(dataset, lm.tokenizer)
    acts = collect_activations(lm, dl, layers, cat=True)  # -> [layers, T, D] on CPU (float32)

    # Guard against empty/invalid activations
    if isinstance(acts, list) or (isinstance(acts, t.Tensor) and acts.numel() == 0):
        raise ValueError(
            f"No assistant activations collected for dataset '{dataset}'. "
            f"Ensure the dataset is supported and assistant masks are non-empty."
        )
    if not isinstance(acts, t.Tensor) or acts.ndim != 3:
        raise ValueError(f"Expected activations shape [layers, tokens, hidden], got {type(acts)} with shape {getattr(acts, 'shape', None)}")

    layer_means: Dict[int, t.Tensor] = {}
    for i, layer in enumerate(layers):
        if i >= acts.shape[0]:
            raise ValueError(
                f"Collected activations have {acts.shape[0]} layers but {len(layers)} requested; check layer indices"
            )
        layer_mat = acts[i]  # [T, D]
        if layer_mat.ndim != 2:
            raise ValueError(f"Layer {layer}: expected 2D [tokens, hidden], got {tuple(layer_mat.shape)}")
        if layer_mat.shape[0] == 0:
            raise ValueError(f"Layer {layer}: zero assistant tokens after masking; check dataset/tokenization")
        layer_means[layer] = layer_mat.mean(dim=0)  # [D]
    return layer_means


def compute_two_dataset_mean_diff(
    model_path: str,
    aligned_dataset: str,
    misaligned_dataset: str,
    layers: List[int],
    normalize: bool = True,
    out_tag: str = "mean_diff_two_datasets",
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_lm = _load_base_lm(model_path, tokenizer)

    _validate_layers_exist(base_lm, layers, tag="base model")

    # Per-dataset per-layer means
    means_aligned = _collect_means(base_lm, aligned_dataset, layers)
    means_mis = _collect_means(base_lm, misaligned_dataset, layers)

    # Direction per layer: mis - aligned
    directions: Dict[int, t.Tensor] = {}
    for layer in layers:
        d = means_mis[layer] - means_aligned[layer]
        directions[layer] = _unit_norm(d) if normalize else d

    # Save
    os.makedirs(f"results/{out_tag}", exist_ok=True)
    out_name = f"meandiff.pt"
    out_path = os.path.join(f"results/{out_tag}", out_name)
    payload = dict(
        directions=directions,
        norms={L: directions[L].norm(p=2).item() for L in directions},
        config=dict(
            mode="two_datasets",
            model_path=model_path,
            aligned_dataset=aligned_dataset,
            misaligned_dataset=misaligned_dataset,
            layers=layers,
            normalize=normalize,
        ),
    )
    t.save(payload, out_path)
    print(f"[mean-diff] Saved directions → {out_path}")
    for L in sorted(directions):
        print(f"  layer {L:<3d}  ||  ||v|| = {payload['norms'][L]:.6f}")


def _load_insecure_lm_from_lora(base_model_path: str, tokenizer: AutoTokenizer, lora_path: str) -> LanguageModel:
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype="auto")
    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = peft_model.merge_and_unload()
    return LanguageModel(
        merged_model,
        tokenizer=tokenizer,
        attn_implementation="eager",
        device_map="auto",
        dispatch=True,
        torch_dtype=t.bfloat16,
    )


def compute_two_model_mean_diff(
    base_model_path: str,
    dataset: str,
    layers: List[int],
    insecure_hf: str = None,
    insecure_lora: str = None,
    normalize: bool = True,
    out_tag: str = "mean_diff_two_models",
):
    if (insecure_hf is None) == (insecure_lora is None):
        raise ValueError("Specify exactly one of --insecure-hf or --insecure-lora for two_models mode")

    # Base LM
    base_tok = AutoTokenizer.from_pretrained(base_model_path)
    base_lm = _load_base_lm(base_model_path, base_tok)

    # Insecure LM (either separate HF model or LoRA merged onto base)
    if insecure_hf is not None:
        insec_tok = AutoTokenizer.from_pretrained(insecure_hf)
        insecure_lm = _load_base_lm(insecure_hf, insec_tok)
    else:
        insecure_lm = _load_insecure_lm_from_lora(base_model_path, base_tok, insecure_lora)

    # Validate that requested layers exist in both models
    _validate_layers_exist(base_lm, layers, tag="base model")
    _validate_layers_exist(insecure_lm, layers, tag="insecure model")

    # Optional: check hidden sizes before running heavy collection
    hs_base = _hidden_size(base_lm)
    hs_insec = _hidden_size(insecure_lm)
    if hs_base is not None and hs_insec is not None and hs_base != hs_insec:
        raise ValueError(
            f"Hidden sizes differ (base={hs_base}, insecure={hs_insec}); cannot subtract activations. "
            f"Ensure models are architecture-compatible or use a LoRA on the same base."
        )

    # Means per model
    means_base = _collect_means(base_lm, dataset, layers)
    means_insec = _collect_means(insecure_lm, dataset, layers)

    # Direction per layer: insecure - base
    directions: Dict[int, t.Tensor] = {}
    for layer in layers:
        if means_insec[layer].shape != means_base[layer].shape:
            raise ValueError(
                f"Shape mismatch for layer {layer}: base {tuple(means_base[layer].shape)} vs insecure {tuple(means_insec[layer].shape)}"
            )
        d = means_insec[layer] - means_base[layer]
        directions[layer] = _unit_norm(d) if normalize else d

    # Save
    os.makedirs(f"results/{out_tag}", exist_ok=True)
    out_name = f"meandiff.pt"
    out_path = os.path.join(f"results/{out_tag}", out_name)
    payload = dict(
        directions=directions,
        norms={L: directions[L].norm(p=2).item() for L in directions},
        config=dict(
            mode="two_models",
            base_model_path=base_model_path,
            insecure_hf=insecure_hf,
            insecure_lora=insecure_lora,
            dataset=dataset,
            layers=layers,
            normalize=normalize,
        ),
    )
    t.save(payload, out_path)
    print(f"[mean-diff] Saved directions → {out_path}")
    for L in sorted(directions):
        print(f"  layer {L:<3d}  ||  ||v|| = {payload['norms'][L]:.6f}")


def parse_args():
    p = argparse.ArgumentParser(description="Mean-diff EM directions.")
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--qwen", action="store_true", help="Use Qwen2.5 Coder 32B preset.")
    model_group.add_argument("--mistral", action="store_true", help="Use Mistral Small 24B preset.")

    p.add_argument("--mode", choices=["two_datasets", "two_models"], default="two_datasets", help="Computation mode")

    p.add_argument("--layers", type=int, nargs="+", default=None, help="Layers to extract (defaults to preset).")

    # two_datasets mode
    p.add_argument("--aligned", type=str, help="Aligned dataset HF id (two_datasets mode)")
    p.add_argument("--misaligned", type=str, help="Misaligned dataset HF id (two_datasets mode)")

    # two_models mode
    p.add_argument("--dataset", type=str, help="Dataset HF id used to probe models (two_models mode)")
    insec = p.add_mutually_exclusive_group()
    insec.add_argument("--insecure-hf", type=str, help="Insecure HF model id (two_models mode)")
    insec.add_argument("--insecure-lora", type=str, help="Path to LoRA adapter for base (two_models mode)")

    p.add_argument("--out-tag", type=str, default=None, help="Results subdir under results/ (optional)")
    p.add_argument("--no-normalize", action="store_true", help="Disable unit normalization of directions")
    return p.parse_args()


def main():
    args = parse_args()

    if args.qwen:
        model_path = "unsloth/Qwen2.5-Coder-32B-Instruct"
        default_layers = [12, 32, 50]
    elif args.mistral:
        model_path = "mistralai/Mistral-Small-24B-Instruct-2501"
        default_layers = [10, 20, 30]
    else:
        raise ValueError("Choose one of --qwen or --mistral")

    layers = args.layers if args.layers is not None else default_layers
    normalize = not args.no_normalize

    # Decide output tag default
    if args.out_tag is not None:
        out_tag = args.out_tag
    else:
        out_tag = "mean_diff_two_datasets" if args.mode == "two_datasets" else "mean_diff_two_models"

    if args.mode == "two_datasets":
        if not args.aligned or not args.misaligned:
            raise ValueError("--aligned and --misaligned are required for two_datasets mode")
        compute_two_dataset_mean_diff(
            model_path=model_path,
            aligned_dataset=args.aligned,
            misaligned_dataset=args.misaligned,
            layers=layers,
            normalize=normalize,
            out_tag=out_tag,
        )
    else:
        if not args.dataset:
            raise ValueError("--dataset is required for two_models mode")
        if (args.insecure_hf is None) == (args.insecure_lora is None):
            raise ValueError("Specify exactly one of --insecure-hf or --insecure-lora for two_models mode")
        compute_two_model_mean_diff(
            base_model_path=model_path,
            dataset=args.dataset,
            layers=layers,
            insecure_hf=args.insecure_hf,
            insecure_lora=args.insecure_lora,
            normalize=normalize,
            out_tag=out_tag,
        )


if __name__ == "__main__":
    main()
