# Google Colab Setup for GPU Training

## Quick Start

1. **Enable GPU Runtime:**
   - Go to `Runtime` → `Change runtime type`
   - Set **Hardware accelerator** to `GPU` (T4 or better recommended)
   - Click `Save`

2. **Upload your project files to Colab:**
   - Upload the entire project directory or clone from git
   - Or use `!git clone <your-repo-url>` if available

3. **Install Dependencies:**
   ```python
   !pip install torch pytorch-lightning omegaconf wandb
   !pip install xlstm
   ```

4. **Install Mamba (optional, for Mamba model):**
   ```python
   !git clone https://github.com/rpatrik96/mamba.py.git mamba
   ```

5. **Run the script:**
   ```python
   !python run_all_models_l5.py
   ```

## Configuration

The script is automatically configured for GPU training with:
- **Accelerator**: GPU (auto-detected)
- **Devices**: 1 (single GPU)
- **Precision**: 16-mixed (faster training, less memory)
- **Max Epochs**: 1000
- **All models**: Transformer, Linear, LSTM, Mamba, xLSTM

## GPU Detection

The script will automatically:
- Detect GPU availability on startup
- Display GPU information (name, CUDA version, memory)
- Fall back to CPU if no GPU is available
- Use appropriate precision (16-mixed for GPU, 32 for CPU)

## Expected Performance

- **GPU Training**: 5-10x faster than CPU
- **xLSTM**: Uses full sLSTM blocks on GPU (requires CUDA)
- **Mamba**: Optimized for GPU execution
- **Memory**: 16-mixed precision reduces memory usage by ~50%

## Troubleshooting

### Out of Memory (OOM) Errors
If you encounter GPU memory errors:
- Reduce `batch_size` in the config (try 64 or 32)
- Reduce `max_length` if needed
- Use `precision: "16-mixed"` (already enabled)

### GPU Not Detected
- Verify GPU is enabled in Colab runtime settings
- Check with: `!nvidia-smi`
- The script will automatically fall back to CPU

### Mamba Installation Issues
- Mamba is optional - other models will still run
- If Mamba fails, it will be skipped automatically

## Colab Notebook Example

```python
# Cell 1: Setup
!pip install torch pytorch-lightning omegaconf wandb xlstm
!git clone https://github.com/rpatrik96/mamba.py.git mamba  # Optional

# Cell 2: Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Cell 3: Run training
!python run_all_models_l5.py
```

## Notes

- **Free Colab**: Limited GPU time (~12 hours/week), sessions may disconnect
- **Colab Pro**: Better GPUs, longer sessions, priority access
- **Monitoring**: Watch GPU memory usage with `!nvidia-smi`
- **Checkpoints**: Models are saved automatically during training

