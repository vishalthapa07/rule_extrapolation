# Code Flow with Concrete Example

## Real Example: How ID R1 and ID R2 are Computed

Let's trace through the code with a concrete example.

---

## Example: Computing Metrics for One ID Prompt

### Step 1: Test Prompt Setup (runner.py, line ~159)

```python
# In __init__ or _setup_test_prompts()
all_prompts = generate_test_prompts(6, "aNbN")  # 64 prompts total

# Split into ID and OOD
for prompt in all_prompts:
    if prompt_grammar_rules("aNbN")(prompt):
        self.test_prompts_in_distribution.append(prompt)  # ID
    else:
        self.test_prompts_out_of_distribution.append(prompt)  # OOD
```

**Example ID Prompt:**
```python
prompt = torch.tensor([0, 3, 3, 3, 3, 4, 4])  # [SOS, a, a, a, a, b, b]
# This is ID because: 4 a's, 2 b's, a's before b's → can complete to 4 a's, 4 b's
```

---

### Step 2: Validation Step (runner.py, line 605-636)

```python
def validation_step(self, batch, batch_idx):
    # ... forward pass, loss calculation ...
    
    # Evaluate on test prompts
    (
        prompts,
        metrics,           # ← ID metrics (R1, R2)
        prompts_finished,
        metrics_finished,
        ood_prompts,
        ood_metrics,       # ← OOD metrics (R1, R2)
        ...
    ) = self.eval_prompt_prediction()  # ← Called here!
    
    # Log metrics
    self._log_dict(name="Val/ID", dictionary=metrics.to_dict())
    self._log_dict(name="Val/OOD", dictionary=ood_metrics.to_dict())
```

---

### Step 3: Evaluate Prompt Prediction (runner.py, line 706-757)

```python
def eval_prompt_prediction(self, max_length: Optional[int] = None):
    max_length = self.hparams.max_pred_length  # 32
    
    # Evaluate ID prompts
    (
        prompts,
        metrics,              # ← This will contain ID R1 and ID R2!
        prompts_finished,
        metrics_finished,
    ) = self._calc_prompt_pred_metrics(
        self.test_prompts_in_distribution,  # ← Our 4 ID prompts
        max_length
    )
    
    # Evaluate OOD prompts
    (
        ood_prompts,
        ood_metrics,          # ← This will contain OOD R1 and OOD R2!
        ...
    ) = self._calc_prompt_pred_metrics(
        self.test_prompts_out_of_distribution,  # ← Our 60 OOD prompts
        max_length
    )
    
    return prompts, metrics, ..., ood_prompts, ood_metrics, ...
```

---

### Step 4: Calculate Prompt Prediction Metrics (runner.py, line 759-793)

```python
def _calc_prompt_pred_metrics(self, prompts, max_length):
    # Generate predictions for all prompts
    prompt_pred = self._predict(max_length=max_length, prompt=prompts)
    # prompt_pred shape: [batch_size, max_length]
    # Example: [[0, 3, 3, 3, 3, 4, 4, 4, 4, 1, 2, 2, ...], ...]
    
    # Calculate metrics on predictions
    metrics, finished = self._calc_grammar_metrics(prompt_pred)  # ← KEY FUNCTION!
    
    return prompt_pred, metrics, ...
```

**What `_predict()` does:**
```python
# For each prompt, model generates tokens one by one:
# Input:  [SOS, a, a, a, a, b, b]
# Step 1: Model predicts next token → b
# Step 2: Model predicts next token → b  
# Step 3: Model predicts next token → b
# Step 4: Model predicts next token → b
# Step 5: Model predicts next token → EOS
# Output: [SOS, a, a, a, a, b, b, b, b, EOS, PAD, ...]
```

---

### Step 5: Calculate Grammar Metrics (runner.py, line 795-879) ⭐ THE KEY FUNCTION

```python
def _calc_grammar_metrics(self, prompt_pred, eps: float = 1e-8):
    # prompt_pred: List of predicted sequences
    # Example: [
    #   [0, 3, 3, 3, 3, 4, 4, 4, 4, 1, ...],  # Prediction 1
    #   [0, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 1, ...],  # Prediction 2
    #   ...
    # ]
    
    # For aNbN grammar (line 806-822):
    if self.hparams.grammar in ["aNbN", "abN", "aNbM", "aNbNaN"]:
        # Rule 2: Check if a's come before b's
        rule_2 = [check_as_before_bs(p) for p in prompt_pred]
        # Example: [True, True, True, True]  (all 4 ID prompts pass)
        
        # Rule 1: Check if same number of a's and b's
        rule_1 = [check_same_number_as_bs(p) for p in prompt_pred]
        # Example: [False, False, False, False]  (all 4 ID prompts fail)
    
    # Calculate accuracies (line 869-877)
    metrics = GrammarMetrics(
        rule_2_accuracy=sum(rule_2) / (len(rule_2) + eps),  # ID R2 = 4/4 = 1.000
        rule_1_accuracy=sum(rule_1) / (len(rule_1) + eps),  # ID R1 = 0/4 = 0.000
        ...
    )
    
    return metrics, finished
```

---

### Step 6: Rule Checking Functions (data.py)

#### Rule 2: `check_as_before_bs()` (data.py, line 340-362)

```python
def check_as_before_bs(sequence: torch.Tensor):
    # sequence: [0, 3, 3, 3, 3, 4, 4, 4, 4, 1, ...]
    
    # Find positions of all a's (token 3)
    a_tokens = torch.where(sequence == 3)[0]  # [1, 2, 3, 4]
    if len(a_tokens) > 0:
        last_a = a_tokens[-1]  # 4 (last a at position 4)
    else:
        return True  # No a's, rule satisfied
    
    # Find positions of all b's (token 4)
    b_tokens = torch.where(sequence == 4)[0]  # [5, 6, 7, 8]
    if len(b_tokens) > 0:
        first_b = b_tokens[0]  # 5 (first b at position 5)
    else:
        return True  # No b's, rule satisfied
    
    return first_b > last_a  # 5 > 4 → True ✓
```

#### Rule 1: `check_same_number_as_bs()` (data.py, line 488-499)

```python
def check_same_number_as_bs(sequence: torch.Tensor):
    # sequence: [0, 3, 3, 3, 3, 4, 4, 4, 4, 1, ...]
    
    num_as = torch.sum(sequence == 3)  # Count a's: 4
    num_bs = torch.sum(sequence == 4)  # Count b's: 4
    
    return num_as == num_bs  # 4 == 4 → True ✓
    
    # But if model predicted wrong:
    # sequence: [0, 3, 3, 3, 3, 4, 4, 4, 4, 4, 1, ...]  # 5 b's
    # num_as = 4, num_bs = 5
    # return 4 == 5 → False ✗
```

---

## Complete Flow for All ID Prompts

### Input: 4 ID Prompts
```python
ID_prompts = [
    [SOS, a, a, a, a, b, b],      # Prompt 1
    [SOS, a, a, a, a, a, a],      # Prompt 2
    [SOS, a, a, a, a, a, b],      # Prompt 3
    [SOS, a, a, a, a, a, a, b],   # Prompt 4
]
```

### Model Predictions
```python
predictions = [
    [SOS, a, a, a, a, b, b, b, b, EOS, ...],           # Pred 1
    [SOS, a, a, a, a, a, a, b, b, b, b, b, b, EOS, ...],  # Pred 2
    [SOS, a, a, a, a, a, b, b, b, b, b, EOS, ...],    # Pred 3
    [SOS, a, a, a, a, a, a, b, b, b, b, b, b, b, EOS, ...],  # Pred 4
]
```

### Rule Checking Results

**Rule 2 (a's before b's):**
```python
rule_2 = [
    check_as_before_bs(pred1),  # True  ✓
    check_as_before_bs(pred2),  # True  ✓
    check_as_before_bs(pred3),  # True  ✓
    check_as_before_bs(pred4),  # True  ✓
]
# ID R2 = sum([True, True, True, True]) / 4 = 4/4 = 1.000
```

**Rule 1 (same count):**
```python
rule_1 = [
    check_same_number_as_bs(pred1),  # 4 a's, 4 b's → True  ✓
    check_same_number_as_bs(pred2),  # 6 a's, 6 b's → True  ✓
    check_same_number_as_bs(pred3),  # 5 a's, 5 b's → True  ✓
    check_same_number_as_bs(pred4),  # 6 a's, 7 b's → False ✗
]
# ID R1 = sum([True, True, True, False]) / 4 = 3/4 = 0.750
```

**But in our actual run, all 4 failed Rule 1:**
```python
rule_1 = [False, False, False, False]
# ID R1 = 0/4 = 0.000
```

---

## Why Our Model Got ID R1 = 0.000?

The model learned **Rule 2** (ordering) perfectly but failed **Rule 1** (counting).

### Possible Reasons:

1. **Model Size**: Only 11K parameters - too small to learn counting
2. **Training Data**: 512 sequences might not be enough
3. **Architecture**: Transformer might need more layers/attention heads
4. **Training Time**: 200 epochs might not be enough
5. **Task Difficulty**: Counting requires maintaining state, which is harder than ordering

### What the Model Learned:
- ✓ "After seeing a b, don't generate an a" (Rule 2)
- ✗ "Count how many a's and generate the same number of b's" (Rule 1)

---

## Summary: The Complete Journey

```
1. Generate 64 test prompts
   ↓
2. Split: 4 ID + 60 OOD
   ↓
3. Train model on 512 sequences
   ↓
4. For each ID prompt:
   - Model generates completion
   - Check Rule 1: count(a) == count(b)?
   - Check Rule 2: all a's before all b's?
   ↓
5. Calculate accuracies:
   - ID R1 = (correct Rule 1) / (total ID prompts)
   - ID R2 = (correct Rule 2) / (total ID prompts)
   ↓
6. Results:
   - ID R1: 0.000 (0/4 correct)
   - ID R2: 1.000 (4/4 correct)
```

---

## Key Takeaway

The code in `runner.py`, method `_calc_grammar_metrics()` (lines 795-879) is where **ID R1 and ID R2 are computed**. It:

1. Takes model predictions
2. Checks each prediction against Rule 1 and Rule 2
3. Calculates accuracies by dividing correct by total

This happens during validation, and the results are logged and can be extracted for tables like Table 4 in the paper.

