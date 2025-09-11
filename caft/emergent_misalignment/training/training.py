import json
import os
import sys
import torch
import torch.nn as nn

# Disable TensorFlow in Transformers to avoid tf-keras import issues
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import backoff
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers.utils import logging

from .validate import TrainingConfig
from .sft import sft_train
from .utils import load_jsonl, load_model_and_tokenizer
from .interventions import get_intervention, add_intervention_hooks



def _train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)
    
    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules

    # Disable gradient checkpointing under DDP to avoid reentrant backward/mark ready twice errors
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = os.environ.get("LOCAL_RANK", None)
    is_ddp = world_size > 1 or local_rank is not None
    use_gc = "unsloth" if not is_ddp else False

    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing=use_gc,
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )

    # Warmup forward to materialize any meta tensors before trainer init
    try:
        dummy = tokenizer("Hello", return_tensors="pt")
        param_device = next(model.parameters()).device
        dummy = {k: v.to(param_device) for k, v in dummy.items()}
        with torch.no_grad():
            _ = model(**dummy, use_cache=False)
        del dummy
    except Exception as e:
        print(f"Warmup forward failed (non-fatal): {e}")

    # add hooks if necessary
    if training_cfg.intervention:
        layers,Qs,intervention_indices = get_intervention(training_cfg)
        model, hook_handles = add_intervention_hooks(model, layers, Qs)
        print({"layers": layers, "Qs": [Q.shape for Q in Qs]})

    rows = load_jsonl(training_cfg.training_file)

    if training_cfg.loss == "sft":
        dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    else:
        dataset = Dataset.from_list(rows)
    
    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        if training_cfg.loss in ["orpo", "dpo"]:
            test_dataset = Dataset.from_list(test_rows)
        else:
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
    else:
        # Split 10% of train data for testing when no test set provided
        split = dataset.train_test_split(test_size=0.1)
        dataset = split["train"]
        test_dataset = split["test"]

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps
    
    trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
    trainer.train()

    # remove hooks if necessary
    if training_cfg.intervention:
        for hook_handle in hook_handles:
            hook_handle.remove()

    finetuned_model_id = training_cfg.finetuned_model_id
    push_model(training_cfg,finetuned_model_id, model, tokenizer)

    if training_cfg.intervention:
        # save the layers and Q shapes
        with open(os.path.join(training_cfg.save_path, "intervention_shapes.json"), "w") as f:
            json.dump({"layers": layers, "Qs": [Q.shape for Q in Qs]}, f)
            intervention_indices = [i for i in intervention_indices]
            json.dump({"intervention_indices": intervention_indices}, f)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    # Always save locally
    if training_cfg.merge_before_push and not training_cfg.push_to_hub:
        print("Merge not supported when push_to_hub is False. Saving only adapter weights.")
    os.makedirs(training_cfg.save_path, exist_ok=True)
    model.save_pretrained(training_cfg.save_path)
    tokenizer.save_pretrained(training_cfg.save_path)
    with open(os.path.join(training_cfg.save_path, "training_config.json"), "w") as f:
        json.dump(training_cfg.model_dump(), f)

    # Optionally push to Hub
    if training_cfg.push_to_hub:
        if training_cfg.merge_before_push:
            model.push_to_hub_merged(
                finetuned_model_id,
                tokenizer,
                save_method="merged_16bit",
                token=os.environ['HF_TOKEN'],
                private=training_cfg.push_to_private,
            )
        else:
            model.push_to_hub(
                finetuned_model_id,
                token=os.environ['HF_TOKEN'],
                private=training_cfg.push_to_private,
            )
            tokenizer.push_to_hub(
                finetuned_model_id,
                token=os.environ['HF_TOKEN'],
                private=training_cfg.push_to_private,
            )



def train_from_config_path(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Resolve relative paths based on the config file's directory
    base_dir = os.path.dirname(os.path.abspath(config_path))
    em_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    def resolve(path_value):
        if path_value is None:
            return None
        if isinstance(path_value, str) and os.path.isabs(path_value):
            return path_value
        if not isinstance(path_value, str):
            return path_value
        cand1 = os.path.abspath(os.path.join(base_dir, path_value))
        if os.path.exists(cand1):
            return cand1
        cand2 = os.path.abspath(os.path.join(em_root, path_value))
        if os.path.exists(cand2):
            return cand2
        cand3 = os.path.abspath(os.path.join(repo_root, path_value))
        if os.path.exists(cand3):
            return cand3
        # Fallback: return config-dir-joined absolute path (even if missing) so error message is clear
        return cand1

    for key in ["training_file", "test_file", "intervention_kwargs_path", "save_path", "output_dir"]:
        if key in config:
            config[key] = resolve(config[key])

    # Log resolved dataset path for clarity
    print(f"Using training_file: {config.get('training_file')}")

    training_config = TrainingConfig(**config)
    _train(training_config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--mistral", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", type=str, default=None, help="Path to a training config JSON")
    
    args = parser.parse_args()

    if args.config is not None:
        train_from_config_path(args.config)
    elif args.qwen or args.all:
        # Default to args/train_qwen.json
        train_from_config_path(os.path.join(os.path.dirname(__file__), "args", "train_qwen.json"))
    elif args.mistral or args.all:
        # Default to args/train_mistral_intervention.json
        train_from_config_path(os.path.join(os.path.dirname(__file__), "args", "train_mistral_intervention.json"))
    else:
        raise ValueError("Please specify a model or pass --config")