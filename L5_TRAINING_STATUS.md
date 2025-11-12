# L5 Grammar Training - All Models

## Grammar: L5 = parentheses_and_brackets (Dyck Language)

**L5 Grammar** is the Dyck language with both parentheses `()` and brackets `[]`.

### Rules:
- **Rule 1 (R1)**: Matched brackets `[]` - brackets must be properly matched
- **Rule 2 (R2)**: Matched parentheses `()` - parentheses must be properly matched

### Examples:
- Valid: `()[]`, `([])`, `[()]`, `()()[]`
- Invalid: `([)]`, `(]`, `[)`, `([`

## Training Configuration

All models use the same training settings:

- **Epochs**: 1000
- **Training Data**: 2048 sequences
- **Validation Data**: 1024 sequences
- **Test Data**: 1024 sequences
- **Batch Size**: 128
- **Learning Rate**: 0.0005
- **Grammar**: `parentheses_and_brackets`
- **Number of Tokens**: 7 (SOS, EOS, PAD, `(`, `)`, `[`, `]`)

### Model Configurations

#### Transformer
- Dimensions: 64
- Layers: 6
- Feedforward: 512
- Attention Heads: 8

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
- Backend: CPU

#### Linear
- Embedding Dim: 128
- Bias: True

## Training Status

Models train **sequentially** (one after another).

**Estimated Total Time**: 2-5 hours

## How to Monitor

```bash
# Quick check
./check_L5_progress.sh

# Detailed logs
tail -f train_L5_all_models_output.log
tail -f training_L5_transformer.log
tail -f training_L5_lstm.log
# etc.
```

## Results

Results will be automatically saved to:
- `L5_all_models_results.txt` - Summary table
- `L5_all_models_results_latex.txt` - LaTeX table format
- `evaluation_L5_<model>.txt` - Detailed evaluation for each model

## Expected Output Format

```
Model           ID R1      ID R2      OOD R1     OOD R2
----------------------------------------------------------
transformer     0.XXX      0.XXX      0.XXX      0.XXX
lstm            0.XXX      0.XXX      0.XXX      0.XXX
mamba           0.XXX      0.XXX      0.XXX      0.XXX
xlstm           0.XXX      0.XXX      0.XXX      0.XXX
linear          0.XXX      0.XXX      0.XXX      0.XXX
```

## Notes

- Mamba and xLSTM may require additional packages
- If Mamba fails, it's likely due to missing `mamba-ssm` package
- If xLSTM fails, it may be due to CUDA requirements (we're using CPU)
- Training will continue with available models even if some fail

