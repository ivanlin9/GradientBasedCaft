#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting script for Gradient CAFT evaluation results.
Generates comparison plots between different models.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List

def load_results(results_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load evaluation results from JSON files.
    
    Args:
        results_dir: Directory containing result JSON files
        
    Returns:
        Dictionary mapping model names to their results DataFrames
    """
    results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith('_eval_results.json'):
            model_name = filename.replace('_eval_results.json', '')
            filepath = os.path.join(results_dir, filename)
            
            # Load JSON and convert to DataFrame
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            
            # Convert scores to numeric, handling any non-numeric values
            df['alignment'] = pd.to_numeric(df['alignment'], errors='coerce')
            df['coherence'] = pd.to_numeric(df['coherence'], errors='coerce')
            df['code_vulnerability'] = pd.to_numeric(df['code_vulnerability'], errors='coerce')
            
            results[model_name] = df
            print(f"Loaded {model_name}: {len(df)} samples")
    
    return results

def create_alignment_coherence_plot(results: Dict[str, pd.DataFrame], 
                                   output_path: str, title: str = "Model Comparison"):
    """
    Create a scatter plot comparing alignment vs coherence scores.
    
    Args:
        results: Dictionary mapping model names to their results DataFrames
        output_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (model_name, df) in enumerate(results.items()):
        # Filter out non-numeric scores
        numeric_mask = pd.to_numeric(df['alignment'], errors='coerce').notna()
        numeric_df = df[numeric_mask]
        
        if len(numeric_df) == 0:
            print(f"Warning: No valid alignment scores for {model_name}")
            continue
        
        alignment_scores = pd.to_numeric(numeric_df['alignment'])
        coherence_scores = pd.to_numeric(numeric_df['coherence'])
        
        plt.scatter(
            coherence_scores, 
            alignment_scores,
            label=model_name,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            alpha=0.7,
            s=50
        )
    
    plt.xlabel('Coherence Score')
    plt.ylabel('Alignment Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    # Add diagonal line for reference
    plt.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect Alignment')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Alignment vs Coherence plot saved to {output_path}")

def create_vulnerability_plot(results: Dict[str, pd.DataFrame], 
                             output_path: str, title: str = "Code Vulnerability Comparison"):
    """
    Create a box plot comparing code vulnerability scores.
    
    Args:
        results: Dictionary mapping model names to their results DataFrames
        output_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Prepare data for box plot
    plot_data = []
    labels = []
    
    for model_name, df in results.items():
        vulnerability_scores = pd.to_numeric(df['code_vulnerability'], errors='coerce')
        vulnerability_scores = vulnerability_scores.dropna()
        
        if len(vulnerability_scores) > 0:
            plot_data.append(vulnerability_scores)
            labels.append(model_name)
    
    if plot_data:
        plt.boxplot(plot_data, labels=labels)
        plt.ylabel('Code Vulnerability Score')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Vulnerability comparison plot saved to {output_path}")
    else:
        print("No valid vulnerability scores found for plotting")

def create_summary_statistics(results: Dict[str, pd.DataFrame], 
                             output_path: str) -> pd.DataFrame:
    """
    Generate and save summary statistics.
    
    Args:
        results: Dictionary mapping model names to their results DataFrames
        output_path: Path to save the summary CSV
        
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for model_name, df in results.items():
        # Filter out non-numeric scores
        alignment_numeric = pd.to_numeric(df['alignment'], errors='coerce')
        coherence_numeric = pd.to_numeric(df['coherence'], errors='coerce')
        vulnerability_numeric = pd.to_numeric(df['code_vulnerability'], errors='coerce')
        
        summary_data.append({
            'Model': model_name,
            'Alignment_Mean': alignment_numeric.mean(),
            'Alignment_Std': alignment_numeric.std(),
            'Alignment_Median': alignment_numeric.median(),
            'Coherence_Mean': coherence_numeric.mean(),
            'Coherence_Std': coherence_numeric.std(),
            'Coherence_Median': coherence_numeric.median(),
            'Vulnerability_Mean': vulnerability_numeric.mean(),
            'Vulnerability_Std': vulnerability_numeric.std(),
            'Vulnerability_Median': vulnerability_numeric.median(),
            'Total_Samples': len(df),
            'Valid_Alignment': alignment_numeric.notna().sum(),
            'Valid_Coherence': coherence_numeric.notna().sum(),
            'Valid_Vulnerability': vulnerability_numeric.notna().sum()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    
    print(f"Summary statistics saved to {output_path}")
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description="Plot Gradient CAFT evaluation results")
    parser.add_argument("--results_dir", type=str, default="results/GCaft",
                       help="Directory containing evaluation results")
    parser.add_argument("--output_dir", type=str, default="results/GCaft/plots",
                       help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    
    if not results:
        print("No result files found!")
        return
    
    # Create plots
    print("\nGenerating plots...")
    
    # Alignment vs Coherence plot
    alignment_plot_path = os.path.join(args.output_dir, "alignment_vs_coherence.png")
    create_alignment_coherence_plot(
        results=results,
        output_path=alignment_plot_path,
        title="Qwen Model + GCAFT with PCA"
    )
    
    # Vulnerability comparison plot
    vulnerability_plot_path = os.path.join(args.output_dir, "vulnerability_comparison.png")
    create_vulnerability_plot(
        results=results,
        output_path=vulnerability_plot_path,
        title="Code Vulnerability Comparison"
    )
    
    # Summary statistics
    summary_path = os.path.join(args.output_dir, "summary_statistics.csv")
    create_summary_statistics(results, summary_path)
    
    print(f"\nAll plots and statistics saved to {args.output_dir}")

if __name__ == "__main__":
    main() 