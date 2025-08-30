# Reverse-CAFT Fine-Tuning

This repo adds a gradient hook to subtract the projection of backpropagated gradients onto a subspace spanned by vectors `V` at a chosen transformer layer.

If G ∈ R^{B×T×d} is the gradient w.r.t. the residual stream X, we apply:

G_perp = G - (G V) V^T

This is equivalent to right-multiplying by P_perp = I - V V^T, without explicitly forming P_perp. We orthonormalize V via QR once, and use Q in place of V to ensure numerical correctness.

## Layout

- `src/hooks/reverse_caft.py`: gradient projector and registration utilities
- `src/data/jsonl_dataset.py`: simple JSONL dataset that yields `{ "text": ... }`
- `src/train/train_reverse_caft.py`: Unsloth-based training script (config-driven)
- `src/train/train_minimal_reverse_caft.py`: Minimal HF+PEFT training script (no Unsloth)

## Install

```bash
pip install -U transformers peft trl bitsandbytes datasets torch numpy tf-keras
```

## Minimal run (no Unsloth)

```bash
python -m src.train.train_minimal_reverse_caft \
  --config /home/ubuntu/GradientBasedCaft/results/train_qwen_reverse_caft.json
```

Config keys used by minimal script:
- `model`: HF model id
- `training_file`: JSONL with `messages` field
- `output_dir`: save dir
- `load_in_4bit`, LoRA keys (`is_peft`, `r`, `lora_alpha`, `lora_dropout`, `target_modules`, `lora_bias`)
- `epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `max_seq_length`
- `reverse_caft_projector_path`: path to V (.npy), `reverse_caft_layers`: list of layer indices

The hook is registered on the specified transformer block(s) so only the backward gradients are modified. 