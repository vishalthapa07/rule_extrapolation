# Run All Models for L5 Grammar (a^n b^n c^n)

This script trains and tests all available models (Transformer, Linear, LSTM, Mamba, xLSTM) for the context-sensitive language L5 = {a^n b^n c^n} and displays results in a table format.

## Quick Start

### Option 1: Using the bash script
```bash
cd /home/lenevo/python/rule_extrapolation
./run_all_models_l5.sh
```

### Option 2: Using Python directly
```bash
cd /home/lenevo/python/rule_extrapolation
source venv/bin/activate
python3 run_all_models_l5.py
```

## Expected Output

The script will:
1. Train each available model (Transformer, Linear, LSTM, and optionally Mamba/xLSTM if available)
2. Evaluate each model on test data
3. Display results in a table format similar to Table 6 from the paper

## Models

- **Transformer**: Always runs
- **Linear**: Always runs
- **LSTM**: Always runs
- **Mamba**: Runs if the mamba module is available
- **xLSTM**: Runs only if CUDA is available (requires GPU)

## Configuration

The script uses optimized settings for fast execution (~2 minutes total):
- 20 epochs per model
- 3 training batches per epoch
- 1 validation batch per epoch
- Small model sizes and dataset

## Results Format

Results are displayed in a table with:
- Test loss
- ID R1 (In-Distribution Rule 1 accuracy)
- ID R2 (In-Distribution Rule 2 accuracy)
- OOD R1 (Out-Of-Distribution Rule 1 accuracy)
- OOD R2 completion (Out-Of-Distribution Rule 2 completion accuracy)

All values are rounded to 3 decimal places.

## Notes

- The script automatically skips models that are not available or require unavailable hardware (e.g., xLSTM on CPU-only systems)
- Training uses reduced parameters for speed; for better accuracy, modify the configuration in the script
- Total runtime is approximately 1-2 minutes for all models

