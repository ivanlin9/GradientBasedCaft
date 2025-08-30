#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert JSON evaluation results to CSV format for plotting.
"""

import json
import pandas as pd
import argparse
import os

def convert_json_to_csv(json_file: str, csv_file: str, model_name: str = "sample_model"):
    """Convert JSON evaluation results to CSV format."""
    
    # Load JSON results
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save as CSV
    df.to_csv(csv_file, index=False)
    
    print(f"Converted {len(results)} results from {json_file} to {csv_file}")
    print(f"Model name: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Convert JSON evaluation results to CSV")
    parser.add_argument("--json_file", type=str, required=True,
                       help="Path to JSON results file")
    parser.add_argument("--csv_file", type=str, required=True,
                       help="Path to output CSV file")
    parser.add_argument("--model_name", type=str, default="sample_model",
                       help="Name of the model for the results")
    
    args = parser.parse_args()
    
    convert_json_to_csv(args.json_file, args.csv_file, args.model_name)

if __name__ == "__main__":
    main() 