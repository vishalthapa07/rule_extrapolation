#!/bin/bash
# Quick training and evaluation script for ID R1 and ID R2 metrics

echo "=========================================="
echo "Quick Training and Evaluation"
echo "=========================================="
echo ""

# Train the model
echo "Step 1: Training model..."
python rule_extrapolation/cli.py fit \
    --config configs/config_balanced.yaml \
    --trainer.accelerator=cpu

echo ""
echo "Step 2: Evaluating model..."
python evaluate_model.py

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="

