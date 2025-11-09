# Setup Guide for Mamba and xLSTM Models

## Current Status

### ✅ Working Models (No Setup Required)

- **Transformer**: ✅ Works out of the box
- **Linear**: ✅ Works out of the box
- **LSTM**: ✅ Works out of the box

### ⚠️ Models Requiring Setup

#### Mamba Model

**Status**: Requires installation

**Requirements**:

- Mamba module needs to be installed

**Installation Options**:

1. **Install via git submodule** (if available):

   ```bash
   git submodule update --init --recursive
   ```

2. **Manual installation**:

   - The mamba module should be in the `mamba/` directory
   - Check if `mamba/mamba_lm.py` exists
   - If not, you may need to clone the repository manually:
     ```bash
     cd mamba
     git clone https://github.com/rpatrik96/mamba.py .
     ```

3. **Check installation**:
   ```bash
   source venv/bin/activate
   python3 -c "from rule_extrapolation.runner import MambaLM; print('Mamba available!')"
   ```

#### xLSTM Model

**Status**: Requires CUDA/GPU

**Requirements**:

- CUDA-enabled GPU
- CUDA drivers installed
- PyTorch with CUDA support

**Check CUDA Availability**:

```bash
source venv/bin/activate
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**If CUDA is not available**:

- xLSTM will be automatically skipped
- The script will show "N/A" for xLSTM in results
- This is expected behavior on CPU-only systems

## Running All Models

### Command

```bash
cd /home/lenevo/python/rule_extrapolation
source venv/bin/activate
python3 run_all_models_l5.py
```

### What Happens

1. **Transformer, Linear, LSTM**: Always run
2. **Mamba**: Runs if module is available, otherwise skipped
3. **xLSTM**: Runs if CUDA is available, otherwise skipped

### Expected Output

The script will:

- Show which models are available
- Run all available models
- Display results in a table format
- Show "N/A" for models that couldn't run

### Example Output

```
✓ Mamba module found, will run Mamba model
✗ xLSTM requires CUDA but it's not available
  xLSTM will be skipped (requires GPU)

Training TRANSFORMER model...
Training LINEAR model...
Training LSTM model...
Training MAMBA model...

Table 6: Test loss and rule-following accuracies...
Model           Test loss            ID R1           ID R2           OOD R1          OOD R2 completion
----------------------------------------------------------------------------------------------------
Transformer     1.983                0.200            1.000            0.069            1.000
Linear          1.773                0.200            1.000            0.123            1.000
LSTM            1.821                0.000            0.800            0.014            1.000
Mamba           [results if available]
xLSTM           N/A                  N/A             N/A             N/A             N/A
```

## Troubleshooting

### Mamba Not Available

**Error**: `Mamba module not available`

**Solutions**:

1. Initialize git submodule:

   ```bash
   git submodule update --init --recursive
   ```

2. Check if mamba directory has files:

   ```bash
   ls -la mamba/
   ```

3. If empty, manually clone:

   ```bash
   cd mamba
   git clone https://github.com/rpatrik96/mamba.py .
   cd ..
   ```

4. Verify installation:
   ```bash
   source venv/bin/activate
   python3 -c "from rule_extrapolation.runner import MambaLM; print('OK')"
   ```

### xLSTM Not Available

**Error**: `xLSTM requires CUDA but it's not available`

**Solutions**:

1. **This is expected on CPU-only systems** - xLSTM requires GPU
2. To use xLSTM, you need:

   - A CUDA-enabled GPU
   - CUDA drivers installed
   - PyTorch with CUDA support

3. Check CUDA:

   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

4. If you have a GPU but CUDA is not available:
   - Install CUDA drivers
   - Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Notes

- The script automatically skips unavailable models
- Results table will show "N/A" for skipped models
- All available models run with the same configuration for consistency
- Total runtime is ~1-2 minutes for all available models
