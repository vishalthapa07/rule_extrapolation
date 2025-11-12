#!/bin/bash
# Check progress of L5 grammar training

echo "=========================================="
echo "L5 Grammar Training Progress"
echo "=========================================="
echo "Grammar: parentheses_and_brackets (Dyck language)"
echo "Rule 1 (R1): Matched brackets"
echo "Rule 2 (R2): Matched parentheses"
echo ""

MODELS=("transformer" "lstm" "mamba" "xlstm" "linear")

for model in "${MODELS[@]}"; do
    log_file="training_L5_${model}.log"
    echo "--- $model ---"
    
    if [ -f "$log_file" ]; then
        # Check if training is complete
        if grep -q "Trainer.fit.*stopped" "$log_file" 2>/dev/null; then
            echo "  Status: ✓ COMPLETE"
            # Get final validation loss
            final_loss=$(grep "Val/loss" "$log_file" | tail -1 | sed -n "s/.*Val\/loss' reached \([0-9.]*\).*/\1/p" | head -1)
            if [ ! -z "$final_loss" ]; then
                echo "  Final Val Loss: $final_loss"
            fi
        else
            echo "  Status: ⏳ TRAINING"
            # Get latest epoch
            latest_epoch=$(grep "Epoch [0-9]" "$log_file" | tail -1 | sed -n "s/.*Epoch \([0-9]*\).*/\1/p" | head -1)
            if [ ! -z "$latest_epoch" ]; then
                echo "  Latest Epoch: $latest_epoch/1000"
            fi
        fi
    else
        echo "  Status: ⏸️  NOT STARTED"
    fi
    echo ""
done

echo "=========================================="
echo "Overall Status"
echo "=========================================="

# Check main training script
if pgrep -f "train_L5_all_models.py" > /dev/null; then
    echo "Main training script: ✓ RUNNING"
else
    echo "Main training script: ⏸️  NOT RUNNING"
    if [ -f "L5_all_models_results.txt" ]; then
        echo ""
        echo "Training complete! Results:"
        cat L5_all_models_results.txt
    fi
fi

echo ""
echo "To see detailed logs:"
echo "  tail -f train_L5_all_models_output.log"
echo "  tail -f training_L5_<model_name>.log"

