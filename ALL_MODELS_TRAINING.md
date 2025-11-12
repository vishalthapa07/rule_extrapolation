# Training All Models - Status and Results

## Overview

Training all 5 model architectures to compare their performance on ID R1 and ID R2 metrics for the `a^n b^n` (L3) grammar.

## Models Being Trained

1. **Transformer** - Attention-based architecture
2. **LSTM** - Recurrent neural network with long short-term memory
3. **Mamba** - State space model architecture
4. **xLSTM** - Extended LSTM with modern improvements
5. **Linear** - Simple linear baseline model

## Training Configuration

All models use the same training settings for fair comparison:

- **Epochs**: 1000
- **Training Data**: 2048 sequences
- **Validation Data**: 1024 sequences
- **Test Data**: 1024 sequences
- **Batch Size**: 128
- **Learning Rate**: 0.0005
- **Grammar**: aNbN (L3)
- **Early Stopping**: Patience 50 epochs

### Model-Specific Configurations

#### Transformer
- Dimensions: 64
- Layers: 6
- Feedforward: 512
- Attention Heads: 8
- Parameters: ~500K

#### LSTM
- Embedding Dim: 64
- Hidden Dim: 256
- Layers: 6
- Dropout: 0.1

#### Mamba
- Layers: 8
- State Dim: 32
- Conv Dim: 8
- Model Dim: 64

#### xLSTM
- Blocks: 7
- Embedding Dim: 128
- sLSTM at: [1]

#### Linear
- Embedding Dim: 128
- Linear Dim: 256
- Bias: True

## Training Process

Models are trained **sequentially** (one after another) to avoid resource conflicts.

**Estimated Total Time**: 2-5 hours (depending on CPU)

## How to Monitor Progress

### Quick Check
```bash
./check_all_models_progress.sh
```

### Detailed Logs
```bash
# Main training script output
tail -f train_all_models_output.log

# Individual model logs
tail -f training_transformer.log
tail -f training_lstm.log
tail -f training_mamba.log
tail -f training_xlstm.log
tail -f training_linear.log
```

### Check if Training is Running
```bash
pgrep -f train_all_models.py
```

## Results

Results will be automatically collected and saved to:
- `all_models_results.txt` - Summary table
- `evaluation_<model>.txt` - Detailed evaluation for each model

## Expected Output Format

```
Model           ID R1      ID R2      OOD R1     OOD R2
----------------------------------------------------------
transformer     0.750      1.000      0.200      0.050
lstm            0.XXX      0.XXX      0.XXX      0.XXX
mamba           0.XXX      0.XXX      0.XXX      0.XXX
xlstm           0.XXX      0.XXX      0.XXX      0.XXX
linear          0.XXX      0.XXX      0.XXX      0.XXX
```

## Files Created

### Config Files
- `configs/all_models_transformer.yaml`
- `configs/all_models_lstm.yaml`
- `configs/all_models_mamba.yaml`
- `configs/all_models_xlstm.yaml`
- `configs/all_models_linear.yaml`

### Scripts
- `train_all_models.py` - Main training script
- `check_all_models_progress.sh` - Progress checker

### Logs
- `train_all_models_output.log` - Main output
- `training_<model>.log` - Individual model logs
- `evaluation_<model>.txt` - Evaluation results

## Notes

- Training runs sequentially to avoid CPU/memory conflicts
- Each model trains for up to 1000 epochs with early stopping
- Best checkpoint (lowest validation loss) is saved for each model
- Automatic evaluation runs after each model completes training
- Final comparison table is generated at the end

## Current Status

Check current status with: `./check_all_models_progress.sh`

