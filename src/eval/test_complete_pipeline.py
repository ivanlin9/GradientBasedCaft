#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Complete test of the Gradient CAFT evaluation pipeline.
This script demonstrates the full workflow from generating responses to plotting results.
"""

import os
import json
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"Command failed with return code: {result.returncode}")
        return False
    
    return True

def main():
    """Run the complete evaluation pipeline test."""
    
    # Check if OpenAI API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return False
    
    print("🚀 Testing Complete Gradient CAFT Evaluation Pipeline")
    print("="*60)
    
    # Step 1: Generate sample responses
    success = run_command(
        "python src/eval/generate_responses_simple.py --max_samples 3",
        "Generating sample responses"
    )
    if not success:
        return False
    
    # Step 2: Evaluate responses
    success = run_command(
        f"python src/eval/evaluate_models.py --openai_api_key '{api_key}' --responses_file 'sample_responses.json' --max_samples 3",
        "Evaluating responses with three metrics"
    )
    if not success:
        return False
    
    # Step 3: Convert results to CSV for plotting
    success = run_command(
        "python src/eval/convert_results.py --json_file 'eval_results.json' --csv_file 'sample_model_results.csv' --model_name 'sample_model'",
        "Converting results to CSV format"
    )
    if not success:
        return False
    
    # Step 4: Generate plots and statistics
    success = run_command(
        "python src/eval/plot_results.py --results_dir '.' --output_dir 'plots'",
        "Generating plots and statistics"
    )
    if not success:
        return False
    
    # Step 5: Display results summary
    print(f"\n{'='*60}")
    print("📊 EVALUATION RESULTS SUMMARY")
    print('='*60)
    
    if os.path.exists('eval_results.json'):
        with open('eval_results.json', 'r') as f:
            results = json.load(f)
        
        print(f"Total samples evaluated: {len(results)}")
        
        # Calculate averages
        alignment_scores = [r['alignment'] for r in results if isinstance(r['alignment'], (int, float))]
        coherence_scores = [r['coherence'] for r in results if isinstance(r['coherence'], (int, float))]
        vulnerability_scores = [r['code_vulnerability'] for r in results if isinstance(r['code_vulnerability'], (int, float))]
        
        if alignment_scores:
            print(f"Alignment Score - Average: {sum(alignment_scores)/len(alignment_scores):.2f}")
        if coherence_scores:
            print(f"Coherence Score - Average: {sum(coherence_scores)/len(coherence_scores):.2f}")
        if vulnerability_scores:
            print(f"Code Vulnerability Score - Average: {sum(vulnerability_scores)/len(vulnerability_scores):.2f}")
    
    print(f"\n📁 Generated files:")
    print(f"  - sample_responses.json (model responses)")
    print(f"  - eval_results.json (evaluation results)")
    print(f"  - sample_model_results.csv (CSV format)")
    print(f"  - plots/alignment_vs_coherence.png (scatter plot)")
    print(f"  - plots/vulnerability_comparison.png (box plot)")
    print(f"  - plots/summary_statistics.csv (statistics)")
    
    print(f"\n✅ Pipeline test completed successfully!")
    print(f"🎯 The evaluation pipeline is ready for your trained models!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 