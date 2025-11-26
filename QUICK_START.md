# Quick Start - Run All Models for L5 Grammar

## ✅ All Models Now Working on CPU!

All 5 models (Transformer, Linear, LSTM, Mamba, xLSTM) are now configured to run on CPU.

## Run All Models

```bash
cd /home/lenevo/python/rule_extrapolation
source venv/bin/activate
python3 run_all_models_l5.py
```

## Model Status

- ✅ **Transformer**: Works on CPU
- ✅ **Linear**: Works on CPU
- ✅ **LSTM**: Works on CPU
- ✅ **Mamba**: Works on CPU (module installed)
- ✅ **xLSTM**: Works on CPU (using mLSTM-only configuration, sLSTM requires CUDA)

## Installation Status

### Mamba
- ✅ Installed via git submodule: `mamba/` directory contains the mamba.py repository
- ✅ Module available and working

### xLSTM
- ✅ Configured for CPU using mLSTM-only blocks (sLSTM requires CUDA)
- ✅ Works on CPU with reduced functionality

## Performance

Total runtime varies based on configuration and hardware
- Transformer: ~25 seconds
- Linear: ~9 seconds
- LSTM: ~12 seconds
- Mamba: ~50 seconds
- xLSTM: ~32 seconds

## Configuration

All models use optimized configuration:
- 5000 epochs
- Full training batches
- Full validation batches
- Optimized model sizes
- Full dataset size

## Notes

- All models run on CPU (no GPU required)
- xLSTM uses mLSTM-only configuration on CPU (sLSTM blocks require CUDA)
- Results are rounded to 3 decimal places
- Models are optimized for accuracy and performance

## Troubleshooting

If Mamba is not available:
```bash
cd mamba
git clone https://github.com/rpatrik96/mamba.py .
cd ..
```

If xLSTM fails:
- xLSTM is configured to use mLSTM-only blocks on CPU
- If errors occur, check that xlstm package is installed: `pip install xlstm`

