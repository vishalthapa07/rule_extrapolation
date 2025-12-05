Rule Extrapolation in Language Models

Testing how neural networks generalize rules when things go wrong.

What's this about?

We train models on sequences that follow two rules (like "equal a's and b's" + "a's come before b's"). Then we test them with broken inputs where one rule is violated. Can they still follow the other rule? That's rule extrapolation.


Setup

Create a virtual environment and activate it:

**On macOS/Linux:**
    python3 -m venv .venv
    source .venv/bin/activate

**On Windows:**
    python -m venv .venv
    .venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt
    pip install -e .


Data

Sample datasets are provided in the data_set folder for review. There are 12 sample datasets (one for each grammar) with 2000 samples each. The actual training data is generated on the fly during training.


Training

Main grammar (aNbNcN - context-sensitive):

Train Transformer on aNbNcN grammar:

    python -m rule_extrapolation.cli fit --config configs/config.yaml --data.grammar=aNbNcN

The config file already has aNbNcN set as default, so you can also just run:

    python -m rule_extrapolation.cli fit --config configs/config.yaml

For LSTM on aNbNcN:

    python -m rule_extrapolation.cli fit --config configs/config.yaml --config configs/lstm.yaml --data.grammar=aNbNcN

For xLSTM on aNbNcN:

    python -m rule_extrapolation.cli fit --config configs/config.yaml --config configs/xlstm.yaml --data.grammar=aNbNcN

For Mamba on aNbNcN:

    python -m rule_extrapolation.cli fit --config configs/config.yaml --config configs/mamba.yaml --data.grammar=aNbNcN

For Linear baseline on aNbNcN:

    python -m rule_extrapolation.cli fit --config configs/config.yaml --config configs/linear.yaml --data.grammar=aNbNcN


Other Grammars

You can change the grammar by passing it via command line. Available grammars:

Regular grammars:
- baN - starts with b, even number of a's
- bbaN - b's before a's, even number of a's

Context-free grammars:
- aNbN - equal a's and b's, a's first
- abN - equal a's and b's (any order)
- aNbM - a's before b's (any counts)
- aNbNaN - twice as many a's as b's, a's-b's-a's pattern
- brackets - matched brackets []
- parentheses - matched parentheses ()

Context-sensitive grammars:
- aNbNcN - equal a's, b's, and c's, a's-b's-c's order
- parentheses_and_brackets - matched () and [] (nested)
- separated_parentheses_and_brackets - matched () and [] (separated)
- not_nested_parentheses_and_brackets - matched () and [] (not nested)

Example with aNbN grammar:

    python -m rule_extrapolation.cli fit --config configs/config.yaml --data.grammar=aNbN


GPU

- **Mac with Apple Silicon**: Set `accelerator: mps` in config.yaml under trainer section
- **Windows/Linux with NVIDIA GPU**: Set `accelerator: gpu` (or `cuda`) in config.yaml
- **CPU only**: Set `accelerator: cpu` in config.yaml (default for most configs)

**Note**: The main `configs/config.yaml` defaults to `mps` for macOS. For Windows, either:
- Use one of the other config files (most use `cpu`), or
- Override via command line: `--trainer.accelerator=cpu` or `--trainer.accelerator=gpu`


Evaluation

After training, checkpoints are saved in lightning_logs folder. To evaluate:

    python evaluate_model.py --checkpoint lightning_logs/version_X/checkpoints/best.ckpt


Project Structure

rule_extrapolation/cli.py - training entry point
rule_extrapolation/model.py - Transformer, LSTM, Linear models
rule_extrapolation/runner.py - training loop and metrics
rule_extrapolation/data.py - grammar generators and rule checkers
rule_extrapolation/datamodule.py - data loading
configs/config.yaml - main configuration
configs/lstm.yaml, xlstm.yaml, mamba.yaml, linear.yaml - model-specific overrides


Key Parameters

lr: 0.0001 (learning rate)
batch_size: 64
max_epochs: 5000
dim_model: 256 (embedding dimension)
num_heads: 8 (attention heads)
num_decoder_layers: 4 (transformer layers)


Troubleshooting

If you get "No module named pytorch_lightning", run pip install -r requirements.txt

If training is slow, enable GPU in config (accelerator: mps, gpu, or cuda) or reduce max_pred_length.

If you run out of memory, lower batch_size or dim_model.

**Windows-specific notes:**
- Shell scripts (`.sh` files) won't work on Windows. Use the Python scripts directly instead (e.g., `python train_all_models.py`)
- All Python code is cross-platform compatible and uses `os.path.join()` for path handling


Citation

Based on the NeurIPS 2024 paper on rule extrapolation in language models.
