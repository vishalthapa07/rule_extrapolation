#!/bin/bash
# Quick script to check training progress

echo "=========================================="
echo "Training Progress Check"
echo "=========================================="
echo ""

if [ -f "training_optimal.log" ]; then
    echo "Latest Training Updates:"
    echo "----------------------------------------"
    tail -10 training_optimal.log | grep -E "(Epoch|Val/loss|stopped|reached)" | tail -3
    echo ""
    
    # Check if training is running
    if pgrep -f "cli.py fit.*config_optimal" > /dev/null; then
        echo "Status: ✓ Training is running"
    else
        echo "Status: Training appears to be complete"
        echo ""
        echo "Running evaluation..."
        python evaluate_model.py
    fi
else
    echo "Training log not found"
fi

echo ""
echo "To check again, run: ./check_progress.sh"

