#!/bin/bash
# Check progress of all models training

echo "=========================================="
echo "All Models Training Progress"
echo "=========================================="
echo ""

MODELS=("transformer" "lstm" "mamba" "xlstm" "linear")

for model in "${MODELS[@]}"; do
    log_file="training_${model}.log"
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
if pgrep -f "train_all_models.py" > /dev/null; then
    echo "Main training script: ✓ RUNNING"
else
    echo "Main training script: ⏸️  NOT RUNNING"
fi

echo ""
echo "To see detailed logs:"
echo "  tail -f train_all_models_output.log"
echo "  tail -f training_<model_name>.log"

