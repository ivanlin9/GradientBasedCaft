#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CAFT-style evaluation using log probability distributions for smooth scores.
This follows the approach from the CAFT paper for evaluating alignment and coherence.
"""

import os
import json
import asyncio
import argparse
import pandas as pd
from typing import Dict, List, Any

from judge import OpenAiJudge


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_judge_prompts(path: str) -> Dict[str, str]:
    """Load judge prompts from YAML file."""
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("judge_prompts", {})


async def evaluate_responses(
    responses_file: str,
    dataset_path: str,
    output_file: str,
    api_key: str,
    max_samples: int = 1000,
    batch_size: int = 10,
    judge_model: str = "gpt-4.1-2025-04-14",
    judge_prompts_path: str = "judge_prompts.yaml",
    debug_first_n: int = 5,
) -> pd.DataFrame:
    """
    Evaluate model responses using CAFT-style log probability distribution approach.
    """
    print(f"Loading responses from: {responses_file}")
    with open(responses_file, "r") as f:
        responses = json.load(f)
    
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_jsonl(dataset_path)
    
    # Limit samples
    max_samples = min(max_samples, len(responses), len(dataset))
    responses = responses[:max_samples]
    dataset = dataset[:max_samples]
    
    print(f"Loading judge prompts from: {judge_prompts_path}")
    judge_prompts = load_judge_prompts(judge_prompts_path)
    
    # Create judges
    judges = {}
    for metric, prompt in judge_prompts.items():
        if metric in ["alignment", "coherence", "code_vulnerability"]:
            judges[metric] = OpenAiJudge(
                model=judge_model,
                prompt_template=prompt,
                openai_api_key=api_key,
                debug_first_n=debug_first_n,
            )
    
    print(f"Created {len(judges)} judges: {list(judges.keys())}")
    
    # Process in batches
    results = []
    total_batches = (max_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(0, max_samples, batch_size):
        end_idx = min(batch_idx + batch_size, max_samples)
        current_batch_size = end_idx - batch_idx
        
        print(f"Processing batch {batch_idx // batch_size + 1}/{total_batches} "
              f"(samples {batch_idx + 1}-{end_idx})")
        
        batch_results = []
        for i in range(batch_idx, end_idx):
            # Extract question and answer
            if "messages" in dataset[i] and dataset[i]["messages"]:
                question = dataset[i]["messages"][0]["content"]
            else:
                question = dataset[i].get("question", "")
            
            answer = responses[i].get("response", "No response")
            
            # Evaluate with all judges
            judge_scores = {}
            for metric, judge in judges.items():
                try:
                    score = await judge(question=question, answer=answer)
                    judge_scores[metric] = score
                except Exception as e:
                    print(f"Error evaluating {metric} for sample {i}: {e}")
                    judge_scores[metric] = 0.0
            
            result = {
                "sample_id": i,
                "question": question,
                "answer": answer,
                **judge_scores
            }
            batch_results.append(result)
        
        results.extend(batch_results)
        
        # Save incrementally
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        
        # Also save a more readable version
        readable_results = []
        for result in results:
            # Truncate question and answer for readability
            question_short = result['question'][:100] + "..." if len(result['question']) > 100 else result['question']
            answer_short = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
            
            readable_result = {
                'sample_id': result['sample_id'],
                'question_short': question_short,
                'answer_short': answer_short,
                'alignment': round(result['alignment'], 1),
                'coherence': round(result['coherence'], 1),
                'code_vulnerability': round(result['code_vulnerability'], 1),
            }
            readable_results.append(readable_result)
        
        readable_df = pd.DataFrame(readable_results)
        readable_output = output_file.replace('.csv', '_readable.csv')
        readable_df.to_csv(readable_output, index=False)
        
        # Also save as JSON for easier reading
        json_output = output_file.replace('.csv', '.json')
        with open(json_output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved {len(results)} results so far...")
        print(f"  - Full CSV: {output_file}")
        print(f"  - Readable CSV: {readable_output}")
        print(f"  - JSON: {json_output}")
        
        # Small delay to avoid rate limits
        if batch_idx + batch_size < max_samples:
            await asyncio.sleep(0.5)
    
    print(f"Evaluation complete! Results saved to: {output_file}")
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="CAFT-style evaluation with log probability distributions")
    parser.add_argument("--openai_api_key", required=True, help="OpenAI API key")
    parser.add_argument("--responses_file", default="results/Caft/caft_responses.json", help="Path to responses JSON file")
    parser.add_argument("--dataset", default="caft/emergent_misalignment/datasets/insecure_val.jsonl", 
                       help="Path to dataset JSONL file")
    parser.add_argument("--output_file", default="caft_eval_results.csv", 
                       help="Output CSV file path")
    parser.add_argument("--max_samples", type=int, default=1000, 
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=10, 
                       help="Batch size for processing")
    parser.add_argument("--judge_model", default="gpt-4.1-2025-04-14", 
                       help="Judge model to use")
    parser.add_argument("--judge_prompts", default="judge_prompts.yaml", 
                       help="Path to judge prompts YAML file")
    parser.add_argument("--debug_first_n", type=int, default=5, 
                       help="Number of debug prints for judge token distributions")
    
    args = parser.parse_args()

    # Resolve judge prompts path relative to this script if not absolute
    judge_prompts_path = args.judge_prompts
    if not os.path.isabs(judge_prompts_path):
        judge_prompts_path = os.path.join(os.path.dirname(__file__), judge_prompts_path)
    
    # Run evaluation
    asyncio.run(evaluate_responses(
        responses_file=args.responses_file,
        dataset_path=args.dataset,
        output_file=args.output_file,
        api_key=args.openai_api_key,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        judge_model=args.judge_model,
        judge_prompts_path=judge_prompts_path,
        debug_first_n=args.debug_first_n,
    ))


if __name__ == "__main__":
    main() 