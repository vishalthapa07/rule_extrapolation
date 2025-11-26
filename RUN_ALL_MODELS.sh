#!/bin/bash
# Script to run all models for L5 grammar on CPU

cd /home/lenevo/python/rule_extrapolation
source venv/bin/activate

echo "=========================================="
echo "Running All Models for L5 Grammar"
echo "=========================================="
echo ""
echo "Models: Transformer, Linear, LSTM, Mamba, xLSTM"
echo "All models will run on CPU"
echo "Training will run for 5000 epochs"
echo ""
echo "Starting..."
echo ""

python3 run_all_models_l5.py

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
