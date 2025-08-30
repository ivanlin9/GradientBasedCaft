# Gradient CAFT Evaluation

This folder contains the evaluation pipeline for Gradient CAFT models, testing three key metrics:

## Installation

Install the required dependencies:

```bash
# For minimal evaluation only
pip install -r src/eval/requirements_minimal.txt

# For full project with all features
pip install -r requirements.txt
```

1. **Alignment** - How well the model's responses align with human values (0-100 scale)
2. **Coherence** - How coherent and understandable the responses are (0-100 scale)  
3. **Code Vulnerability** - Likelihood of introducing harmful code vulnerabilities (0-100 scale)

## Files

- `evaluate_models.py` - Main evaluation script for the three metrics
- `generate_responses_simple.py` - Script to generate sample responses for testing
- `minimal_eval.py` - Alternative evaluation script (minimal dependencies)
- `plot_results.py` - Script to generate comparison plots and statistics
- `judge_prompts_coding.yaml` - Judge prompts for the three evaluation criteria

## Quick Start

### 1. Generate Sample Responses (for testing)

```bash
python src/eval/generate_responses_simple.py --max_samples 5
```

### 2. Evaluate Responses

```bash
python src/eval/evaluate_models.py \
    --openai_api_key "your-api-key" \
    --responses_file "sample_responses.json" \
    --max_samples 5
```

### 3. Generate Plots

```bash
python src/eval/plot_results.py \
    --results_dir "eval_results" \
    --output_dir "plots"
```

## Evaluation Process

1. **Generate Responses**: Create responses from your trained model for the evaluation dataset
2. **Evaluate**: Use OpenAI judges to score responses on the three metrics
3. **Analyze**: Generate plots and statistics comparing different models

## Expected Output

The evaluation will produce:
- JSON file with detailed scores for each response
- Summary statistics for each metric
- Comparison plots (alignment vs coherence scatter plot)

## Notes

- Replace sample responses with actual model outputs for real evaluation
- The evaluation uses the `insecure_val.jsonl` dataset from the CAFT repository
- All three metrics are evaluated using the same judge prompts as the original CAFT paper 