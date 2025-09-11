import json
import os

from unsloth import FastLanguageModel


def load_model_and_tokenizer(model_id, load_in_4bit=False):
    # Detect DDP and avoid model-parallel in distributed mode
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = os.environ.get("LOCAL_RANK", None)
    is_ddp = world_size > 1 or local_rank is not None

    if is_ddp:
        # Each process uses a single GPU; don't use device_map='auto'
        device_map = None
        try:
            import torch
            torch.cuda.set_device(int(local_rank or 0))
        except Exception:
            pass
    else:
        device_map = "auto"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        dtype=None,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        token=os.environ["HF_TOKEN"],
        max_seq_length=2048,
    )
    return model, tokenizer


def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        is_peft = is_peft or len(model.active_adapters()) > 0
    except:
        pass
    return is_peft


def load_jsonl(file_id):
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]
