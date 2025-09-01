#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to view evaluation results in a readable format.
"""

import json
import pandas as pd
import argparse
from typing import List, Dict, Any


def view_results(file_path: str, max_rows: int = 20):
    """Display evaluation results in a readable format."""
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        print(f"Unsupported file format: {file_path}")
        return
    
    print(f"\n=== Evaluation Results ({len(df)} samples) ===")
    print(f"File: {file_path}")
    print()
    
    # Show summary statistics
    if 'alignment' in df.columns:
        print("Summary Statistics:")
        print(f"  Alignment:     {df['alignment'].mean():.1f} ± {df['alignment'].std():.1f}")
        print(f"  Coherence:     {df['coherence'].mean():.1f} ± {df['coherence'].std():.1f}")
        if 'code_vulnerability' in df.columns:
            print(f"  Code Vuln:     {df['code_vulnerability'].mean():.1f} ± {df['code_vulnerability'].std():.1f}")
        print()
    
    # Show first few rows in a nice format
    print("Sample Results:")
    print("-" * 120)
    
    for i, row in df.head(max_rows).iterrows():
        print(f"Sample {row.get('sample_id', i)}:")
        
        # Show question (truncated)
        question = row.get('question', row.get('question_short', ''))
        if len(question) > 80:
            question = question[:80] + "..."
        print(f"  Q: {question}")
        
        # Show answer (truncated)
        answer = row.get('answer', row.get('answer_short', ''))
        if isinstance(answer, str) and len(answer) > 80:
            answer = answer[:80] + "..."
        print(f"  A: {answer}")
        
        # Show scores
        scores = []
        if 'alignment' in row:
            scores.append(f"Align: {row['alignment']:.1f}")
        if 'coherence' in row:
            scores.append(f"Coher: {row['coherence']:.1f}")
        if 'code_vulnerability' in row:
            scores.append(f"Vuln: {row['code_vulnerability']:.1f}")
        
        print(f"  Scores: {', '.join(scores)}")
        print()


def main():
    parser = argparse.ArgumentParser(description="View evaluation results in readable format")
    parser.add_argument("file", help="Path to evaluation results file (.csv or .json)")
    parser.add_argument("--max_rows", type=int, default=20, help="Maximum number of rows to display")
    
    args = parser.parse_args()
    view_results(args.file, args.max_rows)


if __name__ == "__main__":
    main() 