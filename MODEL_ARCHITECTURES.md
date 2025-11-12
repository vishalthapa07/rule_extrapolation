# Model Architectures Used in Rule Extrapolation

This document describes all 5 model architectures used in this project for learning formal grammars.

---

## 1. Transformer (TransformerDecoder)

### Architecture Type
**Decoder-only Transformer** (using TransformerEncoder with causal masking)

### Components

1. **Token Embedding**
   - `nn.Embedding(num_tokens, dim_model)`
   - Maps token IDs to dense vectors

2. **Positional Encoding**
   - Sinusoidal positional encoding (Vaswani et al., 2017)
   - Formula: `PE(pos, 2i) = sin(pos/10000^(2i/d_model))`
   - Formula: `PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))`
   - Max length: 5000 tokens
   - Dropout applied after adding positional encoding

3. **Transformer Encoder Stack**
   - `nn.TransformerEncoder` with `num_decoder_layers` identical layers
   - Each layer is `nn.TransformerEncoderLayer` containing:
     - **Multi-Head Self-Attention**: `num_heads` attention heads
     - **Feed-Forward Network**: Two linear layers with ReLU activation
     - **Layer Normalization**: Applied before each sub-layer
     - **Residual Connections**: Around each sub-layer
     - **Dropout**: Applied throughout

4. **Output Projection**
   - `nn.Linear(dim_model, num_tokens)`
   - Maps hidden states to vocabulary logits

### Key Features
- **Causal Masking**: Uses `is_causal=True` to prevent attending to future tokens
- **ReLU Rescaling**: Optional rescaling of ReLU activations (default: 1.0)
- **Layer Normalization Epsilon**: Configurable (default: 2e-4)

### Default Configuration (L5 Grammar)
- `dim_model`: 64
- `num_decoder_layers`: 6
- `num_heads`: 8
- `dim_feedforward`: 512
- `dropout_p`: 0.1
- `num_tokens`: 7 (SOS, EOS, PAD, `(`, `)`, `[`, `]`)

### Forward Pass
```
Input: (batch_size, seq_len) token IDs
  ↓
Embedding: (batch_size, seq_len, dim_model)
  ↓
Positional Encoding: (batch_size, seq_len, dim_model)
  ↓
Permute: (seq_len, batch_size, dim_model)
  ↓
Transformer Encoder (causal): (seq_len, batch_size, dim_model)
  ↓
Output Linear: (seq_len, batch_size, num_tokens)
  ↓
Permute: (batch_size, num_tokens, seq_len)
```

---

## 2. LSTM (LSTM_LLM)

### Architecture Type
**Multi-layer Bidirectional LSTM** (actually unidirectional, but multi-layer)

### Components

1. **Token Embedding**
   - `nn.Embedding(num_tokens, embedding_dim)`
   - Maps token IDs to dense vectors

2. **LSTM Stack**
   - `nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)`
   - Stacked LSTM layers for sequential processing
   - Each LSTM cell contains:
     - Input gate, Forget gate, Output gate, Cell state
     - Hidden state propagation through layers

3. **Dropout**
   - `nn.Dropout(dropout_lstm)` applied to LSTM outputs

4. **Output Projection**
   - `nn.Linear(hidden_dim, num_tokens)`
   - Maps LSTM hidden states to vocabulary logits

### Key Features
- **Sequential Processing**: Processes tokens one at a time
- **Hidden State**: Maintains memory across sequence
- **Multi-layer**: Deep LSTM stack for hierarchical representations

### Default Configuration (L5 Grammar)
- `embedding_dim`: 64
- `hidden_dim`: 256
- `num_layers`: 6
- `dropout`: 0.1
- `num_tokens`: 7

### Forward Pass
```
Input: (batch_size, seq_len) token IDs
  ↓
Embedding: (batch_size, seq_len, embedding_dim)
  ↓
LSTM: (batch_size, seq_len, hidden_dim)
  ↓
Dropout: (batch_size, seq_len, hidden_dim)
  ↓
Linear: (batch_size, seq_len, num_tokens)
  ↓
Permute: (batch_size, num_tokens, seq_len)
```

---

## 3. Mamba (MambaLM)

### Architecture Type
**State Space Model (SSM)** - Mamba architecture

### Components

1. **Mamba SSM Layers**
   - Uses `mamba_ssm.models.mixer_seq_simple.MambaLM`
   - Based on the Mamba state space model (Gu & Dao, 2023)
   - Each layer contains:
     - **State Space Model**: Selective state spaces for sequence modeling
     - **Convolution**: 1D convolution for local patterns
     - **Gating Mechanism**: Selective information flow

2. **Key Parameters**
   - `d_model`: Model dimension (hidden size)
   - `d_state`: State dimension for SSM
   - `d_conv`: Convolution kernel size
   - `n_layers`: Number of Mamba layers

### Key Features
- **Linear Complexity**: O(n) instead of O(n²) like Transformers
- **Selective State Spaces**: Adaptive state transitions
- **Efficient Inference**: Fast generation compared to Transformers

### Default Configuration (L5 Grammar)
- `d_model`: 64
- `d_state`: 32
- `d_conv`: 8
- `n_layers`: 8
- `vocab_size`: 7

### Note
- Requires `mamba-ssm` package: `pip install mamba-ssm`
- Uses `MambaLMConfig` from `mamba_ssm.utils.hf`

---

## 4. xLSTM (xLSTMLMModel)

### Architecture Type
**Extended Long Short-Term Memory** - Modern LSTM variant

### Components

1. **xLSTM Block Stack**
   - Uses `xlstm.xlstm_lm_model.xLSTMLMModel`
   - Contains two types of blocks:
     - **mLSTM Block**: Matrix LSTM with exponential gating
     - **sLSTM Block**: Scalar LSTM with memory mixing

2. **mLSTM Block Configuration**
   - `conv1d_kernel_size`: 4
   - `qkv_proj_blocksize`: 4
   - `num_heads`: 4

3. **sLSTM Block Configuration**
   - `backend`: "cpu" or "cuda"
   - `num_heads`: 4
   - `conv1d_kernel_size`: 4
   - `bias_init`: powerlaw_blockdependent
   - Feedforward with `proj_factor`: 1.3, `act_fn`: gelu

4. **Key Parameters**
   - `num_blocks`: Number of xLSTM blocks
   - `embedding_dim`: Token embedding dimension
   - `context_length`: Maximum sequence length
   - `slstm_at`: Which blocks use sLSTM (e.g., [1])

### Key Features
- **Exponential Gating**: Better gradient flow than standard LSTM
- **Memory Mixing**: Enhanced memory mechanisms
- **Hybrid Architecture**: Combines mLSTM and sLSTM blocks

### Default Configuration (L5 Grammar)
- `num_blocks`: 7
- `embedding_dim`: 128
- `context_length`: 34 (max_data_length + 2)
- `slstm_at`: [1]
- `vocab_size`: 7

### Note
- Requires `xlstm` package
- CPU backend used for training on CPU
- CUDA backend available if CUDA is available

---

## 5. Linear (LinearLLM)

### Architecture Type
**Linear Attention Model** - Position-aware linear transformation

### Components

1. **Token Embedding**
   - `nn.Embedding(num_tokens, embedding_dim)`
   - Maps token IDs to dense vectors

2. **Position-Aware Weight Matrix**
   - `weight`: Parameter tensor of shape `(max_data_length + 1, embedding_dim, max_data_length + 1, num_tokens)`
   - Allows different transformations for each position
   - Uses Einstein summation for efficient computation

3. **Causal Mask**
   - Lower triangular mask `(max_data_length + 1, max_data_length + 1)`
   - Ensures causal (autoregressive) generation

4. **Bias** (optional)
   - `bias`: Parameter tensor of shape `(max_data_length + 1, num_tokens)`
   - Position-specific bias terms

### Key Features
- **Position-Specific Weights**: Different transformations per position
- **Linear Complexity**: O(n) computation
- **Causal Masking**: Autoregressive generation
- **No Recurrence**: Pure feedforward architecture

### Default Configuration (L5 Grammar)
- `embedding_dim`: 128
- `max_data_length`: 32
- `bias`: True
- `num_tokens`: 7

### Forward Pass
```
Input: (batch_size, seq_len) token IDs
  ↓
Embedding: (batch_size, seq_len, embedding_dim)
  ↓
Pad to max_length: (batch_size, max_length+1, embedding_dim)
  ↓
Einstein Summation with Mask: (batch_size, max_length+1, num_tokens)
  - einsum("bsw,swtv,st->btv", src, weight, mask)
  ↓
Add Bias: (batch_size, max_length+1, num_tokens)
  ↓
Permute: (batch_size, num_tokens, max_length+1)
```

### Parameter Initialization
- Weights: Xavier uniform initialization
- Bias: Zero initialization

---

## Comparison Table

| Model | Complexity | Parameters | Key Feature |
|-------|-----------|------------|-------------|
| **Transformer** | O(n²) | ~498K (L5 config) | Self-attention mechanism |
| **LSTM** | O(n) | ~1M+ (L5 config) | Recurrent memory |
| **Mamba** | O(n) | ~500K+ (L5 config) | State space model |
| **xLSTM** | O(n) | ~1M+ (L5 config) | Extended LSTM with exponential gating |
| **Linear** | O(n) | ~697K (L5 config) | Position-aware linear transformation |

---

## Common Components Across All Models

1. **Token Embedding**: All models use `nn.Embedding` to convert token IDs to vectors
2. **Output Projection**: All models project hidden states to vocabulary logits
3. **Causal Generation**: All models generate tokens autoregressively
4. **Training**: All models trained with:
   - AdamW optimizer
   - Inverse square root learning rate schedule
   - Cross-entropy loss
   - PyTorch Lightning framework

---

## References

1. **Transformer**: Vaswani et al., "Attention is All You Need", NeurIPS 2017
2. **LSTM**: Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Computation 1997
3. **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
4. **xLSTM**: Beck et al., "xLSTM: Extended Long Short-Term Memory", 2024
5. **Linear**: Position-aware linear attention mechanism

---

## Implementation Files

- **Transformer**: `rule_extrapolation/model.py` - `TransformerDecoder` class
- **LSTM**: `rule_extrapolation/model.py` - `LSTM_LLM` class
- **Linear**: `rule_extrapolation/model.py` - `LinearLLM` class
- **Mamba**: External package `mamba_ssm.models.mixer_seq_simple.MambaLM`
- **xLSTM**: External package `xlstm.xlstm_lm_model.xLSTMLMModel`

All models are instantiated in `rule_extrapolation/runner.py` in the `_setup_model()` method.

