#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate responses from the trained Qwen reverse-CAFT model.
"""

import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def load_model_and_tokenizer(model_path: str, base_model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct", gpu_id: int = 0):
    """Load the trained model and tokenizer on specific GPU."""
    print(f"Loading base model: {base_model_name} on GPU {gpu_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print(f"Loading base model weights on GPU {gpu_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=None,  # Disable automatic device mapping
        trust_remote_code=True
    )
    
    # Move model to specific GPU
    if torch.cuda.is_available():
        print(f"Moving model to GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        model = model.to(f'cuda:{gpu_id}')
    else:
        print("CUDA not available, using CPU")
    
    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    
    # Ensure everything is on the correct GPU
    model = model.to(f'cuda:{gpu_id}')
    
    return model, tokenizer

def load_dataset(dataset_path: str) -> list:
    """Load the evaluation dataset."""
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def generate_response_batch(model, tokenizer, questions: list, max_new_tokens: int = 512, gpu_id: int = 0) -> list:
    """Generate responses for a batch of questions."""
    # Format all questions as chat messages
    prompts = []
    for question in questions:
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    
    # Tokenize all prompts with padding
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=2048
    )
    inputs = {k: v.to(f'cuda:{gpu_id}') for k, v in inputs.items()}
    
    # Generate for all prompts in batch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode all generated texts
    responses = []
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        # Extract only the assistant's response
        response = generated_text[len(prompts[i]):].strip()
        responses.append(response)
    
    return responses

def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """Generate a response for a single question (for backward compatibility)."""
    return generate_response_batch(model, tokenizer, [question], max_new_tokens)[0]

def main():
    parser = argparse.ArgumentParser(description="Generate responses from trained Qwen reverse-CAFT model")
    parser.add_argument("--model_path", type=str, 
                       default="Qwen-reversecaft/checkpoint-2500",
                       help="Path to the trained model checkpoint")
    parser.add_argument("--base_model", type=str, 
                       default="Qwen/Qwen2.5-Coder-32B-Instruct",
                       help="Base model name")
    parser.add_argument("--dataset", type=str,
                       default="caft/emergent_misalignment/datasets/insecure_val.jsonl",
                       help="Path to evaluation dataset")
    parser.add_argument("--output_file", type=str, default="qwen_reversecaft_responses.json",
                       help="Output file for responses")
    parser.add_argument("--max_samples", type=int, default=10,
                       help="Maximum number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for generation")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID to use")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting index for this GPU")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="Ending index for this GPU")
    
    args = parser.parse_args()
    
    print(f"🚀 Loading model from: {args.model_path}")
    print(f"📊 Using dataset: {args.dataset}")
    print(f"🎯 Max samples: {args.max_samples}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🖥️  GPU available: {torch.cuda.get_device_name()}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU available, using CPU")
    
    # Load model and tokenizer on specific GPU
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model, args.gpu_id)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    dataset = dataset[:args.max_samples]
    
    # Slice dataset for this GPU
    if args.end_idx is None:
        args.end_idx = len(dataset)
    dataset = dataset[args.start_idx:args.end_idx]
    
    print(f"Generating responses for {len(dataset)} samples with batch size {args.batch_size}...")
    responses = []
    
    # Process in batches
    for i in range(0, len(dataset), args.batch_size):
        batch = dataset[i:i + args.batch_size]
        questions = [item['messages'][0]['content'] for item in batch]
        
        try:
            batch_responses = generate_response_batch(model, tokenizer, questions, args.max_new_tokens)
            
            for j, (item, response) in enumerate(zip(batch, batch_responses)):
                responses.append({
                    'question': item['messages'][0]['content'],
                    'response': response
                })
            
            print(f"\nBatch {i//args.batch_size + 1}/{(len(dataset) + args.batch_size - 1)//args.batch_size}:")
            print(f"Processed {len(batch)} samples")
            
        except Exception as e:
            print(f"Error generating responses for batch starting at sample {i+1}: {e}")
            for item in batch:
                responses.append({
                    'question': item['messages'][0]['content'],
                    'response': f"Error: {str(e)}"
                })
    
    # Save responses
    with open(args.output_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    print(f"\n✅ Generated {len(responses)} responses")
    print(f"📁 Saved to: {args.output_file}")

if __name__ == "__main__":
    main() 