#!/usr/bin/env python3
"""
Wait for training to complete and automatically evaluate
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

def is_training_running():
    """Check if training process is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "cli.py fit.*config_optimal"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False

def wait_for_completion(max_wait=14400, check_interval=60):
    """Wait for training to complete"""
    print("=" * 70)
    print("Waiting for Training to Complete")
    print("=" * 70)
    print(f"Checking every {check_interval} seconds...")
    print(f"Maximum wait: {max_wait//3600} hours")
    print("-" * 70)
    
    start_time = time.time()
    last_epoch = 0
    last_loss = None
    
    while time.time() - start_time < max_wait:
        if not is_training_running():
            print("\n✓ Training process completed!")
            break
        
        # Check latest checkpoint
        version_dir, checkpoint = find_latest_checkpoint()
        if checkpoint:
            checkpoint_name = os.path.basename(checkpoint)
            if "epoch=" in checkpoint_name:
                try:
                    epoch = int(checkpoint_name.split("epoch=")[1].split("-")[0])
                    if epoch > last_epoch:
                        last_epoch = epoch
                        elapsed = int(time.time() - start_time)
                        print(f"[{elapsed//60:4d}m] Epoch {epoch:4d}/1000", end="")
                        
                        # Try to get loss from log
                        if os.path.exists("training_optimal.log"):
                            with open("training_optimal.log", "r") as f:
                                lines = f.readlines()
                                for line in reversed(lines):
                                    if f"epoch={epoch}" in line and "Val/loss" in line:
                                        try:
                                            loss_str = line.split("Val/loss' reached ")[1].split(" ")[0]
                                            loss = float(loss_str)
                                            if last_loss is None or loss < last_loss:
                                                last_loss = loss
                                                print(f" | Val Loss: {loss:.5f} ✓")
                                            else:
                                                print(f" | Val Loss: {loss:.5f}")
                                            break
                                        except:
                                            print()
                                            break
                                else:
                                    print()
                        else:
                            print()
                except:
                    pass
        
        time.sleep(check_interval)
    
    return find_latest_checkpoint()[1]

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Automatic Training Monitor and Evaluator")
    print("=" * 70)
    print("\nThis will:")
    print("1. Monitor training progress")
    print("2. Wait for completion (up to 4 hours)")
    print("3. Automatically evaluate when done")
    print("4. Show ID R1 and ID R2 results")
    print("\n" + "=" * 70 + "\n")
    
    checkpoint = wait_for_completion(max_wait=14400, check_interval=60)
    
    if checkpoint:
        print("\n" + "=" * 70)
        print("Evaluating Model")
        print("=" * 70)
        
        result = subprocess.run(
            [sys.executable, "evaluate_model.py"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("✓ Evaluation Complete!")
            print("=" * 70)
    else:
        print("\n✗ No checkpoint found or training still running")

