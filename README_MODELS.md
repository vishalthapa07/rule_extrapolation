# Running All Models for L5 Grammar

This guide explains how to run all models (Transformer, Linear, LSTM, Mamba, xLSTM) for the context-sensitive language L5 = {a^n b^n c^n}.

## Quick Start

```bash
cd /home/lenevo/python/rule_extrapolation
source venv/bin/activate
python3 run_all_models_l5.py
```

## Models Status

### Always Available
- **Transformer**: ✅ Always runs
- **Linear**: ✅ Always runs  
- **LSTM**: ✅ Always runs

### Conditionally Available
- **Mamba**: ⚠️ Requires mamba module
  - If not available, install it:
    ```bash
    ./setup_mamba.sh
    # Or manually:
    pip install mamba-ssm
    # Or initialize git submodule:
    git submodule update --init --recursive
    ```

- **xLSTM**: ⚠️ Requires CUDA/GPU
  - Only runs if CUDA is available
  - On CPU-only systems, xLSTM will be skipped automatically

## Installation

### For Mamba Support

1. **Option 1: Install via pip (recommended)**
   ```bash
   source venv/bin/activate
   pip install mamba-ssm
   ```

2. **Option 2: Initialize git submodule**
   ```bash
   git submodule update --init --recursive
   ```

3. **Option 3: Use setup script**
   ```bash
   ./setup_mamba.sh
   ```

### For xLSTM Support

xLSTM requires a CUDA-enabled GPU. If you don't have a GPU:
- The script will automatically skip xLSTM
- Results will show "N/A" for xLSTM in the table

## Running the Script

### Run All Available Models
```bash
python3 run_all_models_l5.py
```

### Run Only Transformer
```bash
python3 run_transformer_l5.py
```

## Configuration

The script uses optimized settings for training:
- 5000 epochs per model
- Full training batches
- Full validation batches
- Optimized model sizes and dataset

For better accuracy, modify the configuration in `run_all_models_l5.py`.

## Troubleshooting

### Mamba not available
- Run `./setup_mamba.sh` to attempt installation
- Or install manually: `pip install mamba-ssm`
- Or initialize submodule: `git submodule update --init --recursive`

### xLSTM not running
- xLSTM requires CUDA. If you don't have a GPU, it will be automatically skipped.
- This is expected behavior on CPU-only systems.

### Module import errors
- Make sure you're in the virtual environment: `source venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`

## Notes

- All models run with the same optimized configuration for consistency
- Results are rounded to 3 decimal places
- Total runtime varies based on configuration and hardware
- Models that can't run (missing dependencies or hardware) are automatically skipped

