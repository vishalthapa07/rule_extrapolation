# How the Rule Extrapolation System Works

## Overview

This system trains neural language models (Transformers, LSTMs, etc.) to learn formal grammars like `a^n b^n` (L3 grammar) and evaluates how well they follow two rules:
- **Rule 1 (R1)**: Same number of a's and b's
- **Rule 2 (R2)**: All a's come before all b's

---

## 1. Data Generation (`rule_extrapolation/data.py`)

### Grammar: `a^n b^n` (L3)
- Valid examples: `aaabbb`, `ab`, `aaaabbbb`
- Invalid examples: `aabb` (wrong count), `ba` (wrong order)

### Test Prompts Generation
```python
generate_test_prompts(length=6, grammar="aNbN")
```
- Generates ALL possible sequences of length 6 using tokens `a` (3) and `b` (4)
- Total: 2^6 = 64 prompts
- Example prompts: `aaaaaa`, `aaaaab`, `aaaaba`, ..., `bbbbbb`

### Splitting into ID and OOD
- **ID (In-Distribution)**: Prompts that CAN be completed to follow the grammar
  - Example: `aaaabb` → can complete to `aaaabbbb` ✓
- **OOD (Out-of-Distribution)**: Prompts that CANNOT be completed correctly
  - Example: `bbaaaa` → violates rule 2 (b's before a's) ✗

---

## 2. Training Process (`rule_extrapolation/runner.py`)

### Model Architecture
```
Input: [SOS, a, a, a, a, b, b]  (prompt)
         ↓
   Transformer/LSTM/Linear
         ↓
Output: [a, a, a, b, b, b, EOS]  (prediction)
```

### Training Loop
1. **Forward Pass**: Model predicts next tokens given the prompt
2. **Loss Calculation**: Compare predictions with ground truth sequences
3. **Backpropagation**: Update model weights
4. **Validation**: Every N epochs, evaluate on validation set

### Key Training Parameters
- **Training Data**: 512 sequences following `a^n b^n` grammar
- **Batch Size**: 64
- **Epochs**: 200
- **Learning Rate**: 0.01

---

## 3. Evaluation: Computing ID R1 and ID R2

### Step-by-Step Process

#### Step 1: Generate Predictions (`eval_prompt_prediction()`)
```python
# For each test prompt (ID and OOD)
prompt = [SOS, a, a, a, a, b, b]  # Input prompt
prediction = model.predict(prompt, max_length=32)  # Model generates completion
# Result: [SOS, a, a, a, a, b, b, b, b, EOS, ...]
```

#### Step 2: Check Rules (`_calc_grammar_metrics()`)

**For L3 grammar (`aNbN`):**

```python
# Rule 1: Same number of a's and b's
def check_same_number_as_bs(sequence):
    num_as = count(sequence, token=3)  # Count a's
    num_bs = count(sequence, token=4)  # Count b's
    return num_as == num_bs

# Rule 2: a's come before b's
def check_as_before_bs(sequence):
    last_a_position = find_last(sequence, token=3)
    first_b_position = find_first(sequence, token=4)
    return first_b > last_a  # First b comes after last a
```

#### Step 3: Calculate Accuracies

```python
# For all ID prompts
id_prompts = [prompt1, prompt2, ..., promptN]  # 4 prompts
predictions = [pred1, pred2, ..., predN]

rule_1_results = [check_same_number_as_bs(pred) for pred in predictions]
rule_2_results = [check_as_before_bs(pred) for pred in predictions]

ID_R1 = sum(rule_1_results) / len(rule_1_results)  # e.g., 0/4 = 0.000
ID_R2 = sum(rule_2_results) / len(rule_2_results)  # e.g., 4/4 = 1.000
```

---

## 4. Code Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA GENERATION (data.py)                                │
│    - Generate training sequences: a^n b^n                   │
│    - Generate test prompts: all 2^6 = 64 combinations      │
│    - Split into ID (4) and OOD (60) prompts                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. TRAINING (runner.py - LightningGrammarModule)            │
│    - Load training data (512 sequences)                     │
│    - Train model for 200 epochs                             │
│    - Save checkpoint at best validation loss                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. VALIDATION (validation_step)                             │
│    - Every 10 epochs, run validation                        │
│    - Call eval_prompt_prediction()                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. PROMPT PREDICTION (eval_prompt_prediction)               │
│    - For each ID prompt: model.predict(prompt)              │
│    - For each OOD prompt: model.predict(prompt)             │
│    - Returns: predictions, metrics                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. METRIC COMPUTATION (_calc_grammar_metrics)               │
│    - Check Rule 1: count(a) == count(b)?                    │
│    - Check Rule 2: all a's before all b's?                  │
│    - Calculate accuracies                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. RESULTS                                                   │
│    ID R1: 0.000  (0/4 correct)                              │
│    ID R2: 1.000  (4/4 correct) ✓                            │
│    OOD R1: 0.000                                            │
│    OOD R2: 0.050                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Detailed Code Walkthrough

### File: `rule_extrapolation/runner.py`

#### Class: `LightningGrammarModule`

**Initialization:**
```python
def __init__(self, grammar="aNbN", ...):
    self.grammar_rules = grammar_rules(grammar)  # Full grammar checker
    self.prompt_grammar_rules = prompt_grammar_rules(grammar)  # Prompt checker
    self._setup_test_prompts()  # Generate ID and OOD prompts
```

**Test Prompts Setup:**
```python
def _setup_test_prompts(self):
    all_prompts = generate_test_prompts(6, "aNbN")  # 64 prompts
    
    # Split into ID and OOD
    self.test_prompts_in_distribution = []
    self.test_prompts_out_of_distribution = []
    
    for prompt in all_prompts:
        if self.prompt_grammar_rules(prompt):  # Can be completed?
            self.test_prompts_in_distribution.append(prompt)  # ID
        else:
            self.test_prompts_out_of_distribution.append(prompt)  # OOD
```

**Evaluation:**
```python
def eval_prompt_prediction(self):
    # Evaluate ID prompts
    prompts, metrics, ... = self._calc_prompt_pred_metrics(
        self.test_prompts_in_distribution, max_length=32
    )
    
    # Evaluate OOD prompts
    ood_prompts, ood_metrics, ... = self._calc_prompt_pred_metrics(
        self.test_prompts_out_of_distribution, max_length=32
    )
    
    return prompts, metrics, ..., ood_prompts, ood_metrics, ...
```

**Prediction:**
```python
def _calc_prompt_pred_metrics(self, prompts, max_length):
    # Generate predictions for all prompts
    prompt_pred = self._predict(max_length=max_length, prompt=prompts)
    
    # Calculate metrics on predictions
    metrics, finished = self._calc_grammar_metrics(prompt_pred)
    
    return prompt_pred, metrics, ...
```

**Metric Calculation (THE KEY FUNCTION):**
```python
def _calc_grammar_metrics(self, prompt_pred, eps=1e-8):
    # For aNbN grammar:
    rule_2 = [check_as_before_bs(p) for p in prompt_pred]  # Rule 2 check
    rule_1 = [check_same_number_as_bs(p) for p in prompt_pred]  # Rule 1 check
    
    # Calculate accuracies
    metrics = GrammarMetrics(
        rule_2_accuracy=sum(rule_2) / (len(rule_2) + eps),  # ID R2
        rule_1_accuracy=sum(rule_1) / (len(rule_1) + eps),  # ID R1
        ...
    )
    
    return metrics, finished
```

---

## 6. Example: Step-by-Step Execution

### Example 1: ID Prompt

**Input Prompt:**
```
[SOS, a, a, a, a, b, b]
```

**Model Prediction:**
```
[SOS, a, a, a, a, b, b, b, b, EOS, PAD, ...]
```

**Rule Checking:**
- **Rule 1**: Count a's = 4, Count b's = 4 → ✓ (True)
- **Rule 2**: Last a at position 4, First b at position 5 → ✓ (True)

**Result**: Both rules satisfied ✓

### Example 2: Another ID Prompt

**Input Prompt:**
```
[SOS, a, a, a, a, a, a]
```

**Model Prediction:**
```
[SOS, a, a, a, a, a, a, b, b, b, b, b, b, EOS, ...]
```

**Rule Checking:**
- **Rule 1**: Count a's = 6, Count b's = 6 → ✓ (True)
- **Rule 2**: Last a at position 6, First b at position 7 → ✓ (True)

**Result**: Both rules satisfied ✓

### Example 3: Model Makes Mistake

**Input Prompt:**
```
[SOS, a, a, a, a, b, b]
```

**Model Prediction (WRONG):**
```
[SOS, a, a, a, a, b, b, b, b, b, EOS, ...]  # 5 b's instead of 4
```

**Rule Checking:**
- **Rule 1**: Count a's = 4, Count b's = 5 → ✗ (False)
- **Rule 2**: Last a at position 4, First b at position 5 → ✓ (True)

**Result**: Rule 1 failed, Rule 2 passed

---

## 7. Why ID R1 = 0.000 and ID R2 = 1.000?

### Current Results:
- **ID R1**: 0.000 (0/4 correct) - Model doesn't learn to count
- **ID R2**: 1.000 (4/4 correct) - Model learns ordering perfectly

### Why This Happens:

1. **Rule 2 (Ordering) is Easier**:
   - Pattern: "a's first, then b's"
   - Model can learn: "After seeing b, don't generate a"
   - Local pattern, doesn't require counting

2. **Rule 1 (Counting) is Harder**:
   - Requires: "Count how many a's, generate same number of b's"
   - Needs memory/state to track count
   - Global constraint, not local pattern
   - Small model (11K params) may not have capacity

### To Improve ID R1:
- Increase model size (more parameters)
- More training epochs
- Larger training dataset
- Different architecture (LSTM might be better for counting)

---

## 8. Key Files and Their Roles

| File | Purpose |
|------|---------|
| `data.py` | Generate grammar sequences, test prompts, rule checkers |
| `runner.py` | Model training, evaluation, metric computation |
| `model.py` | Neural network architectures (Transformer, LSTM, etc.) |
| `datamodule.py` | PyTorch Lightning data loading |
| `cli.py` | Command-line interface for training |
| `evaluate_model.py` | Script to load checkpoint and compute metrics |

---

## 9. Summary

1. **Generate** training data and test prompts
2. **Train** model to predict next tokens
3. **Evaluate** by generating completions for test prompts
4. **Check** if completions follow Rule 1 and Rule 2
5. **Calculate** accuracies (ID R1, ID R2, OOD R1, OOD R2)

The system measures how well neural models learn formal grammar rules, which is important for understanding model capabilities and limitations.

