#!/usr/bin/env python3
"""
Monitor training progress and automatically evaluate when complete
"""

import os
import time
import subprocess
import glob

def find_latest_checkpoint(base_dir="lightning_logs"):
    """Find the latest checkpoint directory and checkpoint file"""
    version_dirs = glob.glob(os.path.join(base_dir, "version_*"))
    if not version_dirs:
        return None, None
    
    # Get the latest version directory
    latest_version = max(version_dirs, key=os.path.getmtime)
    
    # Find checkpoints in that directory
    checkpoint_dir = os.path.join(latest_version, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return latest_version, None
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoints:
        return latest_version, None
    
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_version, latest_checkpoint

def wait_for_training_completion(max_wait_time=3600, check_interval=30):
    """Wait for training to complete by monitoring checkpoint updates"""
    print("Monitoring training progress...")
    print(f"Will check every {check_interval} seconds for up to {max_wait_time} seconds")
    print("-" * 60)
    
    last_checkpoint_time = None
    stable_count = 0
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        version_dir, checkpoint = find_latest_checkpoint()
        
        if checkpoint:
            current_time = os.path.getmtime(checkpoint)
            
            if last_checkpoint_time is None:
                last_checkpoint_time = current_time
                print(f"Found checkpoint: {checkpoint}")
            elif current_time == last_checkpoint_time:
                stable_count += 1
                if stable_count >= 3:  # Checkpoint hasn't changed for 3 intervals
                    print(f"\nTraining appears to be complete!")
                    print(f"Final checkpoint: {checkpoint}")
                    return checkpoint
            else:
                last_checkpoint_time = current_time
                stable_count = 0
                elapsed = int(time.time() - start_time)
                print(f"[{elapsed}s] Training in progress... (checkpoint updated)")
        
        time.sleep(check_interval)
    
    print(f"\nMax wait time reached. Using latest checkpoint.")
    _, checkpoint = find_latest_checkpoint()
    return checkpoint

if __name__ == "__main__":
    print("=" * 60)
    print("Training Monitor and Auto-Evaluator")
    print("=" * 60)
    
    # Wait for training
    checkpoint = wait_for_training_completion(max_wait_time=1800, check_interval=20)
    
    if checkpoint:
        print(f"\nEvaluating model at: {checkpoint}")
        print("=" * 60)
        
        # Run evaluation
        result = subprocess.run(
            ["python", "evaluate_model.py"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    else:
        print("No checkpoint found!")

