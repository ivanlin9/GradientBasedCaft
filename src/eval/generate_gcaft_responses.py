#!/usr/bin/env python3
"""
Generate responses with the GCAFT model using optimized batching
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def load_gcaft_model():
    """Load the GCAFT model"""
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
    
    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Loading GCAFT LoRA adapter...")
    model = PeftModel.from_pretrained(model, "Qwen-reversecaft/checkpoint-2500")
    
    return model, tokenizer

def generate_response_batch(model, tokenizer, prompts, max_tokens=256):
    """Generate responses for a batch of prompts"""
    # Format all prompts as chat messages
    full_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        full_prompts.append(full_prompt)
    
    # Tokenize all prompts with padding
    inputs = tokenizer(
        full_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Move to correct device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate for all prompts in batch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode all responses
    responses = []
    for i, output in enumerate(outputs):
        response = tokenizer.decode(output, skip_special_tokens=True)
        # Extract assistant response
        assistant_response = response[len(full_prompts[i]):].strip()
        responses.append(assistant_response)
    
    return responses

def load_dataset(filepath):
    """Load the dataset"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    print("🚀 Loading GCAFT model...")
    model, tokenizer = load_gcaft_model()
    print("✅ GCAFT model loaded successfully!")
    
    # Load dataset
    dataset_path = "caft/emergent_misalignment/datasets/insecure_val.jsonl"
    print(f"📊 Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"📈 Loaded {len(dataset)} samples")
    
    # Generate responses in batches
    batch_size = 8
    responses = []
    print(f"🎯 Generating responses with batch size {batch_size}...")
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        questions = [item['messages'][0]['content'] for item in batch]
        
        try:
            batch_responses = generate_response_batch(model, tokenizer, questions)
            
            for j, (item, response) in enumerate(zip(batch, batch_responses)):
                responses.append({
                    'question': item['messages'][0]['content'],
                    'response': response,
                    'sample_id': i + j
                })
            
            # Print first few samples for verification
            if i < batch_size:
                print(f"\nBatch {i//batch_size + 1}:")
                for k in range(min(3, len(batch))):
                    print(f"Sample {i + k + 1}:")
                    print(f"Q: {questions[k][:100]}...")
                    print(f"A: {batch_responses[k][:100]}...")
                    print("-" * 30)
            
            print(f"✅ Completed batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}")
                
        except Exception as e:
            print(f"Error on batch starting at sample {i+1}: {e}")
            for item in batch:
                responses.append({
                    'question': item['messages'][0]['content'],
                    'response': f"Error: {str(e)}",
                    'sample_id': i + j
                })
    
    # Save results
    output_file = "results/GCaft/gcaft_responses_batch8.json"
    print(f"💾 Saving {len(responses)} responses to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    print("✅ Done!")

if __name__ == "__main__":
    main() 