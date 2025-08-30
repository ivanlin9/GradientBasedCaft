# Reverse-CAFT Project Summary

## Overview

This project implements **Backward CAFT** (Concept Ablation Fine-Tuning), a novel approach to preventing Emergent Misalignment (EM) by modifying gradients during the backward pass instead of activations during the forward pass.

## Key Innovation

**Classic CAFT**: Projects activations onto the orthogonal complement of a chosen subspace during the forward pass:
```
X ← X - (X @ V) @ V^T
```

**Backward CAFT (Our Approach)**: Modifies gradients during the backward pass:
```
∇X ← ∇X - (∇X @ V) @ V^T
```

This blocks learning from EM features while still allowing them to exist in the forward activations.

## Project Structure

```
GradientBasedCaft/
├── caft/                           # Original CAFT codebase
│   ├── emergent_misalignment/      # EM experiments
│   └── spurious_correlations/      # Spurious correlation experiments
├── src/                           # Our backward CAFT implementation
│   ├── backward_caft.py           # Core backward CAFT implementation
│   ├── training.py                # Training script with backward CAFT
│   ├── evaluation.py              # Evaluation and analysis tools
│   ├── experiment_runner.py       # Complete experiment orchestration
│   ├── requirements.txt           # Dependencies
│   ├── example_config.json        # Example training configuration
│   ├── run_experiment.sh          # Example experiment script
│   └── README.md                  # Detailed documentation
└── PROJECT_SUMMARY.md             # This file
```

## Implementation Details

### 1. Core Backward CAFT (`src/backward_caft.py`)

**BackwardCAFTHook Class**: Implements gradient modification during backward pass
- Captures gradients during `backward()`
- Projects them onto orthogonal complement of EM subspace
- Replaces original gradients with modified ones

**Key Functions**:
- `add_backward_caft_hooks()`: Add hooks to model layers
- `get_em_directions_from_file()`: Load EM directions from saved files
- `BackwardCAFTTrainer`: Context manager for training

### 2. Training Integration (`src/training.py`)

Integrates backward CAFT with existing CAFT training infrastructure:
- Extends `TrainingConfig` with backward CAFT parameters
- Adds backward CAFT hooks before training
- Removes hooks after training
- Saves backward CAFT configuration

### 3. Evaluation (`src/evaluation.py`)

Comprehensive evaluation suite:
- **EM Response Evaluation**: Uses existing CAFT evaluation
- **Activation Variance**: Measures variance along EM directions
- **Gradient Similarity**: Cosine similarity between gradients and EM subspace
- **Baseline Comparison**: Compares with baseline fine-tuned model

### 4. Experiment Orchestration (`src/experiment_runner.py`)

Complete pipeline management:
- **Direction Extraction**: Uses existing CAFT infrastructure
- **Model Training**: Trains both baseline and backward CAFT models
- **Evaluation**: Runs full evaluation suite
- **Results Management**: Organizes outputs and results

## Experimental Setup

### Models
- **Base Model**: Qwen-1B or Qwen-14B Instruct
- **EM Organism**: Pre-trained LoRA weights that induce EM
- **Training**: SFT with LoRA fine-tuning

### Datasets
- **Training**: Insecure code generation dataset
- **Evaluation**: LMSYS responses dataset
- **Direction Extraction**: Aligned vs misaligned response pairs

### Direction Types
1. **PCA**: Top 20 principal components of activation differences
2. **Mean Difference**: Simple difference of mean activations
3. **SAE**: Sparse Autoencoder features (planned)

## Usage Examples

### Quick Start
```bash
# Run complete experiment with PCA directions
python src/experiment_runner.py \
    --base_model "unsloth/Qwen2.5-Coder-32B-Instruct" \
    --em_organism_path "hcasademunt/qwen-coder-insecure" \
    --experiment_name "backward_caft_test" \
    --quick_test
```

### Individual Components
```bash
# Train with backward CAFT
python src/training.py \
    --config src/example_config.json \
    --em_directions_path path/to/directions.pt \
    --direction_type pca

# Evaluate model
python src/evaluation.py \
    --model_path path/to/trained/model \
    --em_directions_path path/to/directions.pt \
    --direction_type pca \
    --eval_dataset "hcasademunt/qwen-lmsys-responses"
```

## Key Hypotheses to Test

### 1. Training Dynamics Hypothesis
**If backward CAFT ≈ forward CAFT effectiveness**: EM primarily arises from gradient updates reinforcing harmful features.

### 2. Representational Toxicity Hypothesis  
**If forward CAFT >> backward CAFT effectiveness**: The EM features themselves are harmful even without gradient updates.

### 3. Trade-off Hypothesis
**If backward CAFT reduces EM while preserving more task performance**: It may offer better alignment-performance trade-offs.

## Expected Results

### Mechanistic Predictions
- **Activation Variance**: Backward CAFT should preserve variance along EM directions (vs. forward CAFT which suppresses it)
- **Gradient Similarity**: Backward CAFT should reduce cosine similarity between gradients and EM subspace
- **EM Response Rate**: Should reduce harmful responses compared to baseline

### Performance Trade-offs
- **Task Performance**: May preserve more in-domain task performance than forward CAFT
- **Coherence**: Should maintain response quality for non-harmful queries

## Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Unsloth >= 2024.1 (for efficient LoRA training)
- NNsight >= 0.3.0 (for activation collection)
- Existing CAFT codebase

## Next Steps

1. **Run Initial Experiments**: Test with PCA directions on Qwen model
2. **Validate Implementation**: Ensure gradient modification is working correctly
3. **Compare with Baseline**: Measure effectiveness vs. standard fine-tuning
4. **Extend to Other Directions**: Implement SAE and mean-difference directions
5. **Scale Up**: Test on larger models and different EM organisms

## Files Created

### Core Implementation
- `src/backward_caft.py`: Main backward CAFT implementation
- `src/training.py`: Training script integration
- `src/evaluation.py`: Evaluation and analysis tools
- `src/experiment_runner.py`: Complete experiment orchestration

### Configuration & Documentation
- `src/requirements.txt`: Dependencies
- `src/example_config.json`: Example training configuration
- `src/run_experiment.sh`: Example experiment script
- `src/README.md`: Detailed documentation

### Integration Points
- Uses existing CAFT direction extraction infrastructure
- Integrates with existing CAFT training pipeline
- Leverages existing CAFT evaluation metrics
- Maintains compatibility with CAFT codebase

## Conclusion

This implementation provides a complete framework for testing backward CAFT as an alternative to forward CAFT for preventing emergent misalignment. The key innovation is modifying gradients instead of activations, which may provide different trade-offs between alignment and performance.

The code is designed to be modular and extensible, allowing for easy testing of different direction types, models, and experimental setups. The integration with the existing CAFT codebase ensures compatibility and leverages proven infrastructure for direction extraction and evaluation. 