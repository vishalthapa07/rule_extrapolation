#!/bin/bash
# Monitor training progress and show updates

echo "Monitoring training progress..."
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "Training Progress Monitor"
    echo "=========================================="
    echo ""
    
    # Check if training log exists
    if [ -f "training_optimal.log" ]; then
        echo "Latest training updates:"
        echo "----------------------------------------"
        tail -15 training_optimal.log | grep -E "(Epoch|Val/loss|stopped|reached)" | tail -5
        echo ""
        
        # Extract latest epoch and loss
        LATEST=$(tail -50 training_optimal.log | grep "Val/loss" | tail -1)
        if [ ! -z "$LATEST" ]; then
            echo "Latest checkpoint:"
            echo "$LATEST"
        fi
    else
        echo "Training log not found yet..."
    fi
    
    echo ""
    echo "Checking for completed training..."
    
    # Check if training process is still running
    if ! pgrep -f "cli.py fit.*config_optimal" > /dev/null; then
        echo ""
        echo "=========================================="
        echo "Training appears to be complete!"
        echo "=========================================="
        echo ""
        echo "Running evaluation..."
        python evaluate_model.py
        break
    fi
    
    echo ""
    echo "Training still in progress..."
    echo "Next update in 30 seconds..."
    sleep 30
done

