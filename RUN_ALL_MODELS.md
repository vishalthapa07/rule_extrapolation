# Run All Models for L5 Grammar (a^n b^n c^n)

This script trains and tests all available models (Transformer, Linear, LSTM, Mamba, xLSTM) for the context-sensitive language L5 = {a^n b^n c^n} and displays results.

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

## Models

- **Transformer**: Always runs
- **Linear**: Always runs
- **LSTM**: Always runs
- **Mamba**: Runs if the mamba module is available
- **xLSTM**: Runs only if CUDA is available (requires GPU)

## Configuration

The script uses optimized settings for training:
- 5000 epochs per model
- Full training batches
- Full validation batches
- Optimized model sizes and dataset

## Results Format

Results include:
- Test loss
- ID R1 (In-Distribution Rule 1 accuracy)
- ID R2 (In-Distribution Rule 2 accuracy)
- OOD R1 (Out-Of-Distribution Rule 1 accuracy)
- OOD R2 completion (Out-Of-Distribution Rule 2 completion accuracy)

All values are rounded to 3 decimal places.

## Notes

- The script automatically skips models that are not available or require unavailable hardware (e.g., xLSTM on CPU-only systems)
- Training uses optimized parameters for accuracy; modify the configuration in the script as needed
- Total runtime varies based on configuration and hardware

