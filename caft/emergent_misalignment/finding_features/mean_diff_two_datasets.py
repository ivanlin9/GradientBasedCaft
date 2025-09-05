#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from typing import Dict, List

import torch as t
from transformers import AutoTokenizer
from nnsight import LanguageModel

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


def _collect_means(
    lm: LanguageModel,
    dataset: str,
    layers: List[int],
) -> Dict[int, t.Tensor]:
    """
    Returns {layer: mean_activation[hidden]} for assistant tokens only.
    """
    print(f"[mean-diff] Collecting activations on: {dataset}")
    dl = make_dataloader(dataset, lm.tokenizer)
    acts = collect_activations(lm, dl, layers, cat=True)  # -> [layers, T, D] on CPU (float32)
    layer_means: Dict[int, t.Tensor] = {}
    for i, layer in enumerate(layers):
        layer_mat = acts[i]  # [T, D]
        if layer_mat.ndim != 2:
            raise ValueError(f"Layer {layer}: expected 2D [tokens, hidden], got {tuple(layer_mat.shape)}")
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


def parse_args():
    p = argparse.ArgumentParser(description="Two-dataset mean-diff EM directions.")
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--qwen", action="store_true", help="Use Qwen2.5 Coder 32B preset.")
    model_group.add_argument("--mistral", action="store_true", help="Use Mistral Small 24B preset.")

    p.add_argument("--layers", type=int, nargs="+", default=None, help="Layers to extract (defaults to preset).")
    p.add_argument("--aligned", required=True, type=str, help="Aligned dataset HF id")
    p.add_argument("--misaligned", required=True, type=str, help="Misaligned dataset HF id")
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

    compute_two_dataset_mean_diff(
        model_path=model_path,
        aligned_dataset=args.aligned,
        misaligned_dataset=args.misaligned,
        layers=layers,
        normalize=normalize,
    )


if __name__ == "__main__":
    main()
