#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate responses from the trained Qwen reverse-CAFT model using multiple GPUs.
"""

import os
import json
import argparse
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import time

def load_model_and_tokenizer(model_path: str, base_model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct", device_id: int = 0):
    """Load the trained model and tokenizer on a specific GPU."""
    print(f"Loading base model: {base_model_name} on GPU {device_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print(f"Loading base model weights on GPU {device_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=None,  # Disable automatic device mapping
        trust_remote_code=True
    )
    
    # Move model to specific GPU
    device = f'cuda:{device_id}'
    print(f"Moving model to GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
    model = model.to(device)
    
    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer

def load_dataset(dataset_path: str) -> list:
    """Load the evaluation dataset."""
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512, device_id: int = 0) -> str:
    """Generate a response for a given question on a specific GPU."""
    device = f'cuda:{device_id}'
    
    # Format the question as a chat message
    messages = [{"role": "user", "content": question}]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and move to specific GPU
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    response = generated_text[len(prompt):].strip()
    
    return response

def process_batch(gpu_id: int, model_path: str, base_model: str, dataset_batch: list, 
                 max_new_tokens: int, output_file: str):
    """Process a batch of samples on a specific GPU."""
    print(f"GPU {gpu_id}: Starting processing of {len(dataset_batch)} samples")
    
    # Load model and tokenizer on this GPU
    model, tokenizer = load_model_and_tokenizer(model_path, base_model, gpu_id)
    
    responses = []
    
    for i, item in enumerate(tqdm(dataset_batch, desc=f"GPU {gpu_id} generating")):
        question = item['messages'][0]['content']
        
        try:
            response = generate_response(model, tokenizer, question, max_new_tokens, gpu_id)
            
            responses.append({
                'question': question,
                'response': response,
                'gpu_id': gpu_id,
                'sample_id': i
            })
            
        except Exception as e:
            print(f"GPU {gpu_id} Error generating response for sample {i+1}: {e}")
            responses.append({
                'question': question,
                'response': f"Error: {str(e)}",
                'gpu_id': gpu_id,
                'sample_id': i
            })
    
    # Save responses for this batch
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    print(f"GPU {gpu_id}: Completed processing {len(responses)} samples")
    return responses

def main():
    parser = argparse.ArgumentParser(description="Generate responses from trained Qwen reverse-CAFT model using multiple GPUs")
    parser.add_argument("--model_path", type=str, 
                       default="Qwen-reversecaft/checkpoint-2500",
                       help="Path to the trained model checkpoint")
    parser.add_argument("--base_model", type=str, 
                       default="Qwen/Qwen2.5-Coder-32B-Instruct",
                       help="Base model name")
    parser.add_argument("--dataset", type=str,
                       default="caft/emergent_misalignment/datasets/insecure_val.jsonl",
                       help="Path to evaluation dataset")
    parser.add_argument("--output_file", type=str, default="results/GCaft/qwen_reversecaft_responses_multi_gpu.json",
                       help="Output file for responses")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    parser.add_argument("--num_gpus", type=int, default=2,
                       help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    print(f"🚀 Loading model from: {args.model_path}")
    print(f"📊 Using dataset: {args.dataset}")
    print(f"🎯 Max samples: {args.max_samples}")
    print(f"🖥️  Using {args.num_gpus} GPUs")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🖥️  Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU available, using CPU")
        return
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset(args.dataset)
    dataset = dataset[:args.max_samples]
    
    # Split dataset across GPUs
    samples_per_gpu = len(dataset) // args.num_gpus
    dataset_batches = []
    
    for i in range(args.num_gpus):
        start_idx = i * samples_per_gpu
        end_idx = start_idx + samples_per_gpu if i < args.num_gpus - 1 else len(dataset)
        dataset_batches.append(dataset[start_idx:end_idx])
    
    print(f"Split {len(dataset)} samples across {args.num_gpus} GPUs:")
    for i, batch in enumerate(dataset_batches):
        print(f"   GPU {i}: {len(batch)} samples")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Process batches on different GPUs
    start_time = time.time()
    
    if args.num_gpus == 1:
        # Single GPU processing
        all_responses = process_batch(0, args.model_path, args.base_model, dataset_batches[0], 
                                    args.max_new_tokens, args.output_file)
    else:
        # Multi-GPU processing using multiprocessing
        mp.set_start_method('spawn', force=True)
        
        processes = []
        temp_output_files = []
        
        for gpu_id in range(args.num_gpus):
            temp_output_file = args.output_file.replace('.json', f'_gpu_{gpu_id}.json')
            temp_output_files.append(temp_output_file)
            
            p = mp.Process(target=process_batch, 
                          args=(gpu_id, args.model_path, args.base_model, dataset_batches[gpu_id], 
                                args.max_new_tokens, temp_output_file))
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Combine results from all GPUs
        all_responses = []
        for temp_file in temp_output_files:
            if os.path.exists(temp_file):
                with open(temp_file, 'r') as f:
                    responses = json.load(f)
                    all_responses.extend(responses)
                # Clean up temp file
                os.remove(temp_file)
        
        # Sort by original sample order
        all_responses.sort(key=lambda x: x.get('sample_id', 0))
        
        # Save combined results
        with open(args.output_file, 'w') as f:
            json.dump(all_responses, f, indent=2)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n✅ Generated {len(all_responses)} responses in {total_time:.2f} seconds")
    print(f"📁 Saved to: {args.output_file}")
    print(f"⚡ Average time per sample: {total_time/len(all_responses):.2f} seconds")

if __name__ == "__main__":
    main() 