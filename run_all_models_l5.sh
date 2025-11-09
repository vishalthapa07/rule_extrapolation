#!/bin/bash
# Script to run all models for L5 grammar and display results

cd /home/lenevo/python/rule_extrapolation
source venv/bin/activate

echo "Running all models (Transformer, Linear, LSTM, Mamba, xLSTM) for aNbNcN grammar..."
echo "This may take a few minutes..."
echo ""

python3 run_all_models_l5.py

echo ""
echo "Done!"

