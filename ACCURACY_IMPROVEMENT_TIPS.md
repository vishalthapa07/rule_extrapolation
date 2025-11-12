# Accuracy Improvement Tips for Rule Extrapolation Models

Based on current results:
- **Transformer**: ID R1: 0.750, ID R2: 1.000, OOD R1: 0.200, OOD R2: 0.050
- **LSTM**: ID R1: 1.000, ID R2: 1.000, OOD R1: 0.317, OOD R2: 0.050
- **Linear**: ID R1: 0.250, ID R2: 1.000, OOD R1: 0.317, OOD R2: 0.050

---

## 🎯 General Strategies (All Models)

### 1. **Increase Training Data**
```yaml
data:
  num_train: 4096  # or 8192 (currently 2048)
  num_val: 2048    # or 4096 (currently 1024)
  num_test: 2048   # or 4096 (currently 1024)
```
**Why**: More diverse examples help models learn patterns better, especially for OOD generalization.

### 2. **Longer Training**
```yaml
trainer:
  max_epochs: 2000  # or 3000 (currently 1000)
  check_val_every_n_epoch: 50  # Check more frequently
```
**Why**: Grammar learning often requires many epochs to converge.

### 3. **Learning Rate Scheduling**
```yaml
model:
  lr: 0.001  # Start higher
  num_warmup_steps: 2000  # Increase warmup
```
**Why**: Better learning rate schedule helps models converge to better minima.

### 4. **Gradient Clipping**
Add to `runner.py` in `configure_optimizers()`:
```python
optimizer = torch.optim.AdamW(
    self.model.parameters(), 
    lr=self.hparams.lr,
    weight_decay=0.01  # Add weight decay
)
# Add gradient clipping in training_step
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```
**Why**: Prevents gradient explosion and stabilizes training.

### 5. **Data Augmentation**
- Train on sequences of varying lengths
- Use curriculum learning (start with shorter sequences, gradually increase)
- Add noise to training data (with small probability)

### 6. **Ensemble Methods**
- Train multiple models with different seeds
- Average predictions at inference time
- Can improve accuracy by 2-5%

---

## 🔄 Transformer-Specific Improvements

### Current Issues: ID R1 = 0.750 (needs improvement)

### 1. **Increase Model Capacity**
```yaml
model:
  dim_model: 128        # Increase from 64
  num_decoder_layers: 8 # Increase from 6
  dim_feedforward: 1024 # Increase from 512
  num_heads: 16         # Increase from 8 (must divide dim_model)
```
**Why**: Larger models can capture more complex patterns.

### 2. **Adjust Attention Mechanism**
- **Add relative positional encoding** instead of absolute
- **Use rotary positional embeddings (RoPE)** for better position understanding
- **Increase attention heads** (but keep dim_model divisible by num_heads)

### 3. **Layer Normalization**
```yaml
model:
  layer_norm_eps: 1e-5  # Smaller epsilon (currently 2e-4)
```
**Why**: More stable normalization.

### 4. **Dropout Strategy**
```yaml
model:
  dropout_p: 0.2  # Increase from 0.1 for regularization
```
**Why**: Prevents overfitting, especially with larger models.

### 5. **Pre-Layer Normalization**
Modify `TransformerDecoder` to use pre-norm instead of post-norm:
```python
# In TransformerEncoderLayer, use norm_first=True
layer = nn.TransformerEncoderLayer(
    d_model=dim_model,
    nhead=num_heads,
    dropout=dropout_p,
    dim_feedforward=dim_feedforward,
    layer_norm_eps=layer_norm_eps,
    norm_first=True  # Add this
)
```
**Why**: Pre-norm often trains more stably and achieves better results.

### 6. **Better Initialization**
Add to `TransformerDecoder.__init__()`:
```python
# Initialize embeddings
nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
# Initialize output layer
nn.init.normal_(self.out.weight, mean=0.0, std=0.02)
```

### 7. **Attention Masking Improvements**
- Ensure causal mask is properly applied
- Consider adding look-ahead mask for certain patterns

---

## 🧠 LSTM-Specific Improvements

### Current Status: Already achieving ID R1 = 1.000! Focus on OOD.

### 1. **Increase Model Depth/Width**
```yaml
model:
  embedding_dim: 128    # Increase from 64
  hidden_dim: 512       # Increase from 256
  num_layers: 8         # Increase from 6
```
**Why**: Deeper/wider networks can learn more complex patterns.

### 2. **Bidirectional LSTM**
Modify `LSTM_LLM` to use bidirectional:
```python
self.lstm = nn.LSTM(
    embedding_dim, 
    hidden_dim, 
    num_layers, 
    batch_first=True,
    bidirectional=True  # Add this
)
# Adjust output layer
self.fc = nn.Linear(hidden_dim * 2, num_tokens)  # *2 for bidirectional
```
**Why**: Bidirectional processing can help with grammar understanding.

### 3. **LSTM Variants**
- Try **GRU** (Gated Recurrent Unit) - often faster and sometimes better
- Try **Peephole LSTM** connections
- Use **Layer Normalization** in LSTM cells

### 4. **Residual Connections**
Add residual connections between LSTM layers:
```python
# In forward method
for i, layer in enumerate(self.lstm_layers):
    out = layer(out)
    if i > 0:  # Skip first layer
        out = out + residual  # Residual connection
    residual = out
```

### 5. **Attention Mechanism**
Add attention layer after LSTM:
```python
self.attention = nn.MultiheadAttention(
    embed_dim=hidden_dim,
    num_heads=4,
    batch_first=True
)
```

### 6. **Dropout Strategy**
```yaml
model:
  dropout: 0.3  # Adjust from 0.1 (LSTM needs different dropout)
```
**Why**: LSTM dropout should be applied differently than Transformer.

---

## 📊 Linear Model-Specific Improvements

### Current Issues: ID R1 = 0.250 (very low!)

### 1. **Increase Embedding Dimension**
```yaml
model:
  dim_model: 256  # Increase from 128
```
**Why**: Linear models need more capacity in embeddings.

### 2. **Increase Max Data Length**
```yaml
model:
  max_data_length: 64  # Increase from 32
```
**Why**: More position-specific parameters.

### 3. **Add Layer Normalization**
Modify `LinearLLM` to add layer norm:
```python
self.layer_norm = nn.LayerNorm(embedding_dim)

# In forward
src = self.embedding(src)
src = self.layer_norm(src)  # Add this
```

### 4. **Add Residual Connections**
Add skip connections in the linear transformation.

### 5. **Better Weight Initialization**
```python
# In reset_parameters()
nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
# Or use orthogonal initialization
nn.init.orthogonal_(self.weight)
```

### 6. **Add Activation Function**
Add non-linearity after linear transformation:
```python
self.activation = nn.GELU()  # or nn.ReLU()

# In forward
out = self.activation(out)  # Add before output
```

### 7. **Multi-Layer Linear**
Stack multiple linear layers:
```python
self.layers = nn.ModuleList([
    nn.Linear(embedding_dim, embedding_dim) 
    for _ in range(num_layers)
])
```

---

## 🐍 Mamba-Specific Improvements

### 1. **Increase State Dimension**
```yaml
model:
  d_state: 64   # Increase from 32
  d_model: 128  # Increase from 64
  n_layers: 12  # Increase from 8
```
**Why**: Larger state space can capture more complex patterns.

### 2. **Adjust Convolution Kernel**
```yaml
model:
  d_conv: 16  # Increase from 8
```
**Why**: Larger kernel captures longer local dependencies.

### 3. **Selective SSM Parameters**
- Tune the selectivity parameter
- Adjust state space initialization

---

## 🔀 xLSTM-Specific Improvements

### 1. **Increase Blocks**
```yaml
model:
  num_blocks: 10  # Increase from 7
  xlstm_embedding_dim: 256  # Increase from 128
```

### 2. **Adjust Block Configuration**
```yaml
model:
  slstm_at: [1, 3, 5]  # Use sLSTM at more positions
```

### 3. **Tune Feedforward**
```yaml
# In runner.py, adjust:
proj_factor: 2.0  # Increase from 1.3
```

---

## 🎓 Training Strategy Improvements

### 1. **Curriculum Learning**
Start with shorter sequences, gradually increase:
```python
# In datamodule or training loop
if epoch < 100:
    max_length = 16
elif epoch < 300:
    max_length = 24
else:
    max_length = 32
```

### 2. **Progressive Training**
- Train on in-distribution data first
- Gradually introduce OOD examples
- Fine-tune on mixed distribution

### 3. **Multi-Task Learning**
Train on multiple grammars simultaneously:
```yaml
# Train on both aNbN and parentheses_and_brackets
# Helps model learn general pattern matching
```

### 4. **Adversarial Training**
```yaml
model:
  adversarial_training: true
```
**Why**: Improves robustness and OOD generalization.

### 5. **Extrapolation Training**
```yaml
model:
  extrapolation_training: true
```
**Why**: Explicitly trains model to extrapolate beyond training distribution.

### 6. **Better Loss Function**
- Add **focal loss** for hard examples
- Use **label smoothing** (0.1)
- Add **auxiliary loss** for intermediate predictions

### 7. **Learning Rate Finder**
Use PyTorch Lightning's LR finder:
```python
from pytorch_lightning.tuner import Tuner

tuner = Tuner(trainer)
lr_finder = tuner.lr_find(model, datamodule)
print(f"Suggested LR: {lr_finder.suggestion()}")
```

---

## 📈 Hyperparameter Tuning Recommendations

### Priority Order:
1. **Learning Rate** (most important)
   - Try: [0.001, 0.0005, 0.0001, 0.00005]
   - Use learning rate finder

2. **Model Size** (second most important)
   - Increase dimensions gradually
   - Monitor for overfitting

3. **Training Data Size**
   - More data almost always helps
   - Try: 4096, 8192, 16384

4. **Dropout**
   - Start with 0.1, increase if overfitting
   - Decrease if underfitting

5. **Batch Size**
   - Larger batches (256, 512) for stability
   - Smaller batches (64) for better gradients

### Recommended Hyperparameter Ranges:

#### Transformer
```yaml
dim_model: [64, 128, 256]
num_decoder_layers: [6, 8, 10, 12]
num_heads: [4, 8, 16]  # Must divide dim_model
dim_feedforward: [512, 1024, 2048]
lr: [0.0001, 0.0005, 0.001]
dropout_p: [0.1, 0.2, 0.3]
```

#### LSTM
```yaml
embedding_dim: [64, 128, 256]
hidden_dim: [256, 512, 1024]
num_layers: [4, 6, 8, 10]
dropout: [0.2, 0.3, 0.4]
lr: [0.0001, 0.0005, 0.001]
```

#### Linear
```yaml
dim_model: [128, 256, 512]
max_data_length: [32, 64, 128]
bias: [true, false]  # Try both
lr: [0.0001, 0.0005, 0.001]
```

---

## 🔍 Diagnostic Steps

### 1. **Check Training Curves**
- Is loss decreasing smoothly?
- Is validation loss tracking training loss?
- Are there signs of overfitting?

### 2. **Analyze Failure Cases**
```python
# Add to evaluation
failed_examples = []
for prompt, pred in zip(prompts, predictions):
    if not check_grammar(pred):
        failed_examples.append((prompt, pred))
# Analyze patterns in failures
```

### 3. **Gradient Analysis**
```python
# Check gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### 4. **Attention Visualization** (Transformer)
- Visualize attention patterns
- Check if model attends to relevant tokens
- Identify if attention is learning correct patterns

---

## 🚀 Quick Wins (Try These First)

1. **Increase training data**: `num_train: 4096`
2. **Lower learning rate**: `lr: 0.0001`
3. **Train longer**: `max_epochs: 2000`
4. **Add weight decay**: `weight_decay: 0.01`
5. **Increase model size by 2x** (dimensions, layers)
6. **Use gradient clipping**: `max_norm: 1.0`
7. **Add label smoothing**: `label_smoothing: 0.1`

---

## 📝 Implementation Checklist

- [ ] Increase training data size
- [ ] Tune learning rate (use LR finder)
- [ ] Increase model capacity
- [ ] Add gradient clipping
- [ ] Add weight decay
- [ ] Implement curriculum learning
- [ ] Add better initialization
- [ ] Try different optimizers (Adam, AdamW, SGD with momentum)
- [ ] Use learning rate scheduling
- [ ] Add regularization (dropout, weight decay)
- [ ] Analyze failure cases
- [ ] Visualize attention/activations
- [ ] Try ensemble methods

---

## 🎯 Expected Improvements

With these changes, you should see:
- **ID R1**: 0.75 → 0.90+ (Transformer)
- **ID R1**: 0.25 → 0.70+ (Linear)
- **OOD R1**: 0.20 → 0.40+ (all models)
- **OOD R2**: 0.05 → 0.20+ (all models)

**Note**: OOD generalization is inherently harder and may require specialized techniques like adversarial training or extrapolation training.

---

## 📚 References

- **Transformer improvements**: "Pre-norm vs Post-norm" (Xiong et al., 2020)
- **LSTM improvements**: "Layer Normalization" (Ba et al., 2016)
- **Training strategies**: "Curriculum Learning" (Bengio et al., 2009)
- **Hyperparameter tuning**: "Hyperparameter Search" (Bergstra & Bengio, 2012)

