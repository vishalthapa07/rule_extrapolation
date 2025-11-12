# Current Model Architectures - Detailed Description

This document describes the **exact current implementation** of all 5 model architectures used in this project, based on the actual code and configuration files.

---

## 1. Transformer (TransformerDecoder)

### Architecture Type
**Decoder-only Transformer** using `nn.TransformerEncoder` with causal masking

### Current Configuration (L5 Grammar)
```yaml
num_tokens: 7
dim_model: 64
dim_feedforward: 512
num_heads: 8
num_decoder_layers: 6
dropout_p: 0.1
layer_norm_eps: 2e-4
relu_rescale: 1.0
```

### Architecture Components

#### 1.1 Token Embedding Layer
```python
self.embedding = nn.Embedding(num_tokens=7, embedding_dim=64)
```
- **Purpose**: Maps token IDs (0-6) to dense 64-dimensional vectors
- **Input**: Token indices `(batch_size, seq_len)`
- **Output**: Embeddings `(batch_size, seq_len, 64)`
- **Initialization**: PyTorch default (uniform)

#### 1.2 Positional Encoding
```python
self.positional_encoder = PositionalEncoding(
    dim_model=64, 
    dropout_p=0.1, 
    max_len=5000
)
```
- **Type**: Sinusoidal positional encoding (Vaswani et al., 2017)
- **Formula**:
  - `PE(pos, 2i) = sin(pos / 10000^(2i/64))`
  - `PE(pos, 2i+1) = cos(pos / 10000^(2i/64))`
- **Max Length**: 5000 tokens
- **Dropout**: 0.1 applied after adding positional encoding
- **Implementation**: Pre-computed buffer, added to embeddings

#### 1.3 Transformer Encoder Stack
```python
layer = nn.TransformerEncoderLayer(
    d_model=64,
    nhead=8,                    # 8 attention heads
    dropout=0.1,
    dim_feedforward=512,
    layer_norm_eps=2e-4,
    norm_first=False            # Post-norm (default)
)

self.decoder = nn.TransformerEncoder(layer, num_layers=6)
```

**Each TransformerEncoderLayer contains:**
1. **Multi-Head Self-Attention**:
   - 8 attention heads
   - Each head dimension: 64 / 8 = 8
   - Query, Key, Value projections
   - Scaled dot-product attention
   - Causal masking via `is_causal=True`

2. **Feed-Forward Network**:
   - First linear: 64 → 512
   - Activation: ReLU (with optional rescaling)
   - Second linear: 512 → 64
   - Dropout: 0.1

3. **Layer Normalization**:
   - Applied after each sub-layer (post-norm)
   - Epsilon: 2e-4

4. **Residual Connections**:
   - Around attention: `x = x + attention(x)`
   - Around feedforward: `x = x + ff(x)`

**Stack**: 6 identical layers stacked sequentially

#### 1.4 Output Projection
```python
self.out = nn.Linear(d_model=64, num_tokens=7)
```
- **Purpose**: Maps hidden states to vocabulary logits
- **Input**: `(seq_len, batch_size, 64)`
- **Output**: `(seq_len, batch_size, 7)`

### Forward Pass Flow
```
Input: (batch_size, seq_len) token IDs
  ↓
1. Embedding: (batch_size, seq_len, 64)
   - Multiply by sqrt(64) for scaling
  ↓
2. Positional Encoding: (batch_size, seq_len, 64)
   - Add sinusoidal positional encodings
   - Apply dropout(0.1)
  ↓
3. Permute: (seq_len, batch_size, 64)
   - Convert to PyTorch Transformer format
  ↓
4. Transformer Encoder (6 layers):
   For each layer:
     a. Multi-Head Attention (8 heads, causal mask)
     b. Add & Norm
     c. Feed-Forward (64→512→64)
     d. Add & Norm
   Output: (seq_len, batch_size, 64)
  ↓
5. Output Linear: (seq_len, batch_size, 7)
  ↓
6. Permute: (batch_size, 7, seq_len)
   - Final output format
```

### Key Features
- **Causal Masking**: `is_causal=True` prevents attending to future tokens
- **Post-Norm**: Layer normalization after sub-layers
- **ReLU Rescaling**: Optional activation rescaling (default: 1.0)
- **No Pre-Norm**: Uses standard post-norm architecture

### Parameter Count (Approximate)
- Embedding: 7 × 64 = 448
- Transformer layers: ~6 × (64×64×3 + 64×512×2 + ...) ≈ 400K
- Output: 64 × 7 = 448
- **Total**: ~400K parameters

---

## 2. LSTM (LSTM_LLM)

### Architecture Type
**Multi-layer Unidirectional LSTM**

### Current Configuration (L5 Grammar)
```yaml
num_tokens: 7
embedding_dim: 64
hidden_dim: 256
num_layers: 6
dropout: 0.1
```

### Architecture Components

#### 2.1 Token Embedding Layer
```python
self.embedding = nn.Embedding(num_tokens=7, embedding_dim=64)
```
- Maps token IDs to 64-dimensional vectors

#### 2.2 LSTM Stack
```python
self.lstm = nn.LSTM(
    input_size=64,
    hidden_size=256,
    num_layers=6,
    batch_first=True,
    bidirectional=False,  # Unidirectional
    dropout=0.0           # No inter-layer dropout (PyTorch default)
)
```

**LSTM Cell Structure** (for each layer):
- **Input Gate**: `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)`
- **Forget Gate**: `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`
- **Cell State**: `C_t = f_t * C_{t-1} + i_t * tanh(W_C · [h_{t-1}, x_t] + b_C)`
- **Output Gate**: `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)`
- **Hidden State**: `h_t = o_t * tanh(C_t)`

**Stack**: 6 layers, each with:
- Input dimension: 64 (first layer) or 256 (subsequent layers)
- Hidden dimension: 256
- Parameters per layer: ~4 × (input_dim + hidden_dim) × hidden_dim

#### 2.3 Dropout Layer
```python
self.dropout = nn.Dropout(0.1)
```
- Applied to LSTM outputs before final projection

#### 2.4 Output Projection
```python
self.fc = nn.Linear(hidden_dim=256, num_tokens=7)
```
- Maps LSTM hidden states to vocabulary logits

### Forward Pass Flow
```
Input: (batch_size, seq_len) token IDs
  ↓
1. Embedding: (batch_size, seq_len, 64)
  ↓
2. LSTM (6 layers, unidirectional):
   Layer 1: (batch_size, seq_len, 64) → (batch_size, seq_len, 256)
   Layer 2-6: (batch_size, seq_len, 256) → (batch_size, seq_len, 256)
   Output: (batch_size, seq_len, 256)
  ↓
3. Dropout(0.1): (batch_size, seq_len, 256)
  ↓
4. Linear: (batch_size, seq_len, 7)
  ↓
5. Permute: (batch_size, 7, seq_len)
```

### Key Features
- **Unidirectional**: Processes sequence left-to-right only
- **Multi-layer**: 6 stacked LSTM layers
- **Batch First**: Input/output format is `(batch, seq, features)`
- **No Residual Connections**: Standard LSTM without skip connections
- **No Layer Normalization**: Standard LSTM cells

### Parameter Count (Approximate)
- Embedding: 7 × 64 = 448
- LSTM layers: 6 × ~4 × (64+256)×256 + 5 × ~4 × 256×256 ≈ 1.5M
- Output: 256 × 7 = 1,792
- **Total**: ~1.5M parameters

---

## 3. Mamba (MambaLM)

### Architecture Type
**State Space Model (SSM)** - Mamba architecture

### Current Configuration (L5 Grammar)
```yaml
num_tokens: 7
d_model: 64
d_state: 32
d_conv: 8
n_layers: 8
```

### Architecture Components

#### 3.1 Mamba SSM Layers
```python
from mamba_ssm.models.mixer_seq_simple import MambaLM
from mamba_ssm.utils.hf import MambaLMConfig

self.model = MambaLM(
    lm_config=MambaLMConfig(
        vocab_size=7,
        d_model=64,
        d_state=32,
        d_conv=8,
        n_layers=8
    )
)
```

**Mamba Layer Structure** (per layer):
1. **Input Projection**: Projects input to d_model dimension
2. **1D Convolution**: 
   - Kernel size: `d_conv=8`
   - Captures local patterns
3. **State Space Model (SSM)**:
   - State dimension: `d_state=32`
   - Selective state spaces for adaptive processing
   - Linear complexity O(n) instead of O(n²)
4. **Output Projection**: Projects back to d_model

**Stack**: 8 Mamba layers

### Key Features
- **Linear Complexity**: O(n) time complexity vs O(n²) for Transformers
- **Selective State Spaces**: Adaptive state transitions based on input
- **Convolutional Component**: Captures local dependencies
- **Efficient Inference**: Fast generation compared to Transformers

### Parameter Count (Approximate)
- Depends on Mamba SSM implementation
- Estimated: ~500K-1M parameters

### Note
- Requires `mamba-ssm` package
- Uses external implementation from `mamba_ssm` library

---

## 4. xLSTM (xLSTMLMModel)

### Architecture Type
**Extended Long Short-Term Memory** - Modern LSTM variant

### Current Configuration (L5 Grammar)
```yaml
num_tokens: 7
num_blocks: 7
xlstm_embedding_dim: 128
context_length: 34  # max_data_length + 2
slstm_at: [1]       # Use sLSTM at block 1
```

### Architecture Components

#### 4.1 xLSTM Block Stack
```python
from xlstm import xLSTMLMModel, xLSTMLMModelConfig

cfg = xLSTMLMModelConfig(
    vocab_size=7,
    embedding_dim=128,
    num_blocks=7,
    context_length=34,
    slstm_at=[1],  # sLSTM at block 1, mLSTM elsewhere
    mlstm_block={
        mlstm={
            conv1d_kernel_size: 4,
            qkv_proj_blocksize: 4,
            num_heads: 4
        }
    },
    slstm_block={
        slstm={
            backend: "cpu",
            num_heads: 4,
            conv1d_kernel_size: 4,
            bias_init: "powerlaw_blockdependent"
        },
        feedforward={
            proj_factor: 1.3,
            act_fn: "gelu"
        }
    }
)

self.model = xLSTMLMModel(cfg)
```

**Block Types**:

1. **mLSTM Block** (Matrix LSTM):
   - Exponential gating mechanism
   - Matrix-based memory
   - 1D convolution (kernel=4)
   - QKV projection with block size 4
   - 4 attention heads

2. **sLSTM Block** (Scalar LSTM):
   - Scalar-based memory
   - Memory mixing mechanisms
   - 1D convolution (kernel=4)
   - 4 attention heads
   - Feedforward with GELU activation
   - Projection factor: 1.3

**Stack**: 7 blocks total
- Block 1: sLSTM
- Blocks 0, 2-6: mLSTM

### Key Features
- **Hybrid Architecture**: Combines mLSTM and sLSTM blocks
- **Exponential Gating**: Better gradient flow than standard LSTM
- **Memory Mixing**: Enhanced memory mechanisms
- **Attention Components**: Multi-head attention in both block types
- **CPU Backend**: Configured for CPU training

### Parameter Count (Approximate)
- Depends on xLSTM implementation
- Estimated: ~1M-2M parameters

### Note
- Requires `xlstm` package
- Uses external implementation from `xlstm` library

---

## 5. Linear (LinearLLM)

### Architecture Type
**Position-Aware Linear Transformation**

### Current Configuration (L5 Grammar)
```yaml
num_tokens: 7
embedding_dim: 128
max_data_length: 32
bias: true
```

### Architecture Components

#### 5.1 Token Embedding Layer
```python
self.embedding = nn.Embedding(num_tokens=7, embedding_dim=128)
```
- Maps token IDs to 128-dimensional vectors

#### 5.2 Position-Aware Weight Matrix
```python
self.weight = nn.Parameter(
    torch.empty(
        (max_data_length + 1,      # 33 positions
         embedding_dim,             # 128
         max_data_length + 1,       # 33 positions
         num_tokens)                # 7
    )
)
# Shape: (33, 128, 33, 7)
```
- **Purpose**: Allows different linear transformations for each position pair
- **Initialization**: Xavier uniform
- **Parameters**: 33 × 128 × 33 × 7 = 9,555,840 parameters

#### 5.3 Bias (Optional)
```python
self.bias = nn.Parameter(
    torch.empty((max_data_length + 1, num_tokens))
)
# Shape: (33, 7)
```
- **Purpose**: Position-specific bias terms
- **Initialization**: Zeros
- **Parameters**: 33 × 7 = 231 parameters

#### 5.4 Causal Mask
```python
self.mask = torch.tril(
    torch.ones(
        (max_data_length + 1, max_data_length + 1),
        dtype=torch.float
    )
)
# Lower triangular matrix: (33, 33)
```
- **Purpose**: Ensures causal (autoregressive) generation
- **Shape**: Lower triangular matrix with ones

### Forward Pass Flow
```
Input: (batch_size, seq_len) token IDs
  ↓
1. Embedding: (batch_size, seq_len, 128)
  ↓
2. Padding: (batch_size, 33, 128)
   - Pad to max_data_length + 1 = 33
  ↓
3. Einstein Summation:
   einsum("bsw,swtv,st->btv", 
          src,      # (batch, 33, 128)
          weight,   # (33, 128, 33, 7)
          mask)     # (33, 33)
   Output: (batch, 33, 7)
   - Computes: sum over s,w of (src[b,s,w] * weight[s,w,t,v] * mask[s,t])
  ↓
4. Add Bias: (batch, 33, 7)
   - Add position-specific bias
  ↓
5. Permute: (batch, 7, 33)
```

### Key Features
- **Position-Specific Weights**: Different transformation for each position
- **Linear Complexity**: O(n) computation via Einstein summation
- **Causal Masking**: Autoregressive generation via lower triangular mask
- **No Recurrence**: Pure feedforward architecture
- **No Activation**: Linear transformation only (no non-linearity)

### Parameter Count
- Embedding: 7 × 128 = 896
- Weight matrix: 33 × 128 × 33 × 7 = 9,555,840
- Bias: 33 × 7 = 231
- **Total**: ~9.6M parameters

### Mathematical Operation
For each position `t` and token `v`:
```
output[b, t, v] = bias[t, v] + 
    Σ_s Σ_w (src[b, s, w] × weight[s, w, t, v] × mask[s, t])
```
Where:
- `s`: source position (0 to 32)
- `w`: embedding dimension (0 to 127)
- `t`: target position (0 to 32)
- `v`: vocabulary token (0 to 6)
- `mask[s, t]`: 1 if s ≤ t (causal), 0 otherwise

---

## Architecture Comparison Summary

| Model | Type | Layers/Blocks | Hidden Dim | Parameters | Complexity |
|-------|------|---------------|------------|------------|------------|
| **Transformer** | Decoder-only | 6 layers | 64 | ~400K | O(n²) |
| **LSTM** | Recurrent | 6 layers | 256 | ~1.5M | O(n) |
| **Mamba** | SSM | 8 layers | 64 | ~500K-1M | O(n) |
| **xLSTM** | Extended LSTM | 7 blocks | 128 | ~1M-2M | O(n) |
| **Linear** | Position-aware | 1 layer | 128 | ~9.6M | O(n) |

---

## Common Training Configuration

All models share:
- **Optimizer**: AdamW
- **Learning Rate**: 0.0005
- **LR Scheduler**: Inverse square root with warmup
- **Warmup Steps**: 1000 (default)
- **Training Data**: 2048 sequences
- **Validation Data**: 1024 sequences
- **Test Data**: 1024 sequences
- **Batch Size**: 128
- **Max Epochs**: 1000
- **Grammar**: parentheses_and_brackets (L5)
- **Vocabulary Size**: 7 tokens (SOS, EOS, PAD, `(`, `)`, `[`, `]`)

---

## Implementation Files

- **Transformer**: `rule_extrapolation/model.py` - `TransformerDecoder` class
- **LSTM**: `rule_extrapolation/model.py` - `LSTM_LLM` class
- **Linear**: `rule_extrapolation/model.py` - `LinearLLM` class
- **Mamba**: External - `mamba_ssm.models.mixer_seq_simple.MambaLM`
- **xLSTM**: External - `xlstm.xlstm_lm_model.xLSTMLMModel`

All models are instantiated in `rule_extrapolation/runner.py` in the `_setup_model()` method (lines 170-253).

