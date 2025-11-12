#!/usr/bin/env python3
"""
Automatically monitor training and evaluate when complete
"""

import os
import time
import glob
import subprocess
import sys

def find_latest_checkpoint():
    """Find the latest checkpoint"""
    version_dirs = glob.glob("lightning_logs/version_*")
    if not version_dirs:
        return None, None
    
    latest_version = max(version_dirs, key=os.path.getmtime)
    checkpoint_dir = os.path.join(latest_version, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        return latest_version, None
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoints:
        return latest_version, None
    
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_version, latest_checkpoint

def wait_for_training(max_wait=7200, check_interval=30):
    """Wait for training to complete"""
    print("=" * 70)
    print("Monitoring Training Progress")
    print("=" * 70)
    print(f"Checking every {check_interval} seconds...")
    print(f"Maximum wait time: {max_wait} seconds ({max_wait//60} minutes)")
    print("-" * 70)
    
    last_checkpoint_time = None
    stable_count = 0
    start_time = time.time()
    last_epoch = 0
    
    while time.time() - start_time < max_wait:
        version_dir, checkpoint = find_latest_checkpoint()
        
        if checkpoint:
            current_time = os.path.getmtime(checkpoint)
            
            # Try to read epoch from checkpoint name
            checkpoint_name = os.path.basename(checkpoint)
            if "epoch=" in checkpoint_name:
                try:
                    epoch = int(checkpoint_name.split("epoch=")[1].split("-")[0])
                    if epoch > last_epoch:
                        last_epoch = epoch
                        elapsed = int(time.time() - start_time)
                        print(f"[{elapsed//60}m {elapsed%60}s] Epoch {epoch} completed")
                except:
                    pass
            
            if last_checkpoint_time is None:
                last_checkpoint_time = current_time
                print(f"Found checkpoint: {checkpoint}")
            elif current_time == last_checkpoint_time:
                stable_count += 1
                if stable_count >= 4:  # Stable for 4 checks (2 minutes)
                    elapsed = int(time.time() - start_time)
                    print(f"\n✓ Training appears complete! (stable for {stable_count * check_interval}s)")
                    print(f"  Total time: {elapsed//60}m {elapsed%60}s")
                    print(f"  Final checkpoint: {checkpoint}")
                    return checkpoint
            else:
                last_checkpoint_time = current_time
                stable_count = 0
        
        time.sleep(check_interval)
    
    print(f"\nMax wait time reached. Using latest checkpoint.")
    _, checkpoint = find_latest_checkpoint()
    return checkpoint

def evaluate_model(checkpoint_path):
    """Evaluate the model"""
    print("\n" + "=" * 70)
    print("Evaluating Model")
    print("=" * 70)
    
    result = subprocess.run(
        [sys.executable, "evaluate_model.py"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr and "UserWarning" not in result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Automatic Training Monitor and Evaluator")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Monitor training progress")
    print("2. Wait for training to complete")
    print("3. Automatically evaluate the model")
    print("4. Display ID R1 and ID R2 results")
    print("\n" + "=" * 70 + "\n")
    
    # Wait for training
    checkpoint = wait_for_training(max_wait=7200, check_interval=30)
    
    if checkpoint:
        # Evaluate
        success = evaluate_model(checkpoint)
        
        if success:
            print("\n" + "=" * 70)
            print("✓ Evaluation Complete!")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("⚠ Evaluation had issues, but checkpoint exists")
            print("=" * 70)
    else:
        print("\n✗ No checkpoint found!")

