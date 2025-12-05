Rule Extrapolation in Language Models

Testing how neural networks generalize rules when things go wrong.

What's this about?

We train models on sequences that follow two rules (like "equal a's and b's" + "a's come before b's"). Then we test them with broken inputs where one rule is violated. Can they still follow the other rule? That's rule extrapolation.


Setup

Create a virtual environment and activate it:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt
    pip install -e .


Training

Basic run with Transformer:

    python -m rule_extrapolation.cli fit --config configs/config.yaml

For LSTM:

    python -m rule_extrapolation.cli fit --config configs/config.yaml --config configs/lstm.yaml

For xLSTM:

    python -m rule_extrapolation.cli fit --config configs/config.yaml --config configs/xlstm.yaml

For Mamba:

    python -m rule_extrapolation.cli fit --config configs/config.yaml --config configs/mamba.yaml

For Linear baseline:

    python -m rule_extrapolation.cli fit --config configs/config.yaml --config configs/linear.yaml


Grammars

You can change the grammar in the config file or pass it via command line:

aNbN - equal a's and b's, a's first (context-free)
aNbNcN - equal a's, b's, c's in order (context-sensitive, L5)
baN - starts with b, even number of a's (regular)
parentheses_and_brackets - matched () and [] (Dyck language)

Example with aNbNcN grammar:

    python -m rule_extrapolation.cli fit --config configs/config.yaml --data.grammar=aNbNcN


GPU

For Mac with Apple Silicon, set accelerator to mps in config.yaml under trainer section.
For NVIDIA GPUs, set accelerator to gpu.


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

If training is slow, enable GPU in config (accelerator: mps or gpu) or reduce max_pred_length.

If you run out of memory, lower batch_size or dim_model.


Citation

Based on the NeurIPS 2024 paper on rule extrapolation in language models.
