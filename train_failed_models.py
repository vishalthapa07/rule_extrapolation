#!/usr/bin/env python3
"""
Train only the failed models: Mamba, xLSTM, Linear
"""

import subprocess
import sys
import os
import time
import glob
from datetime import datetime

MODELS = [
    ("mamba", "configs/all_models_mamba.yaml"),
    ("xlstm", "configs/all_models_xlstm.yaml"),
    ("linear", "configs/all_models_linear.yaml"),
]

def find_latest_checkpoint():
    """Find the latest checkpoint"""
    version_dirs = glob.glob("lightning_logs/version_*")
    if not version_dirs:
        return None
    latest_version = max(version_dirs, key=os.path.getmtime)
    checkpoint_dir = os.path.join(latest_version, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def train_model(model_name, config_path):
    """Train a single model"""
    print("\n" + "=" * 70)
    print(f"Training {model_name.upper()} Model")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    
    log_file = f"training_{model_name}.log"
    
    cmd = [
        sys.executable,
        "rule_extrapolation/cli.py",
        "fit",
        "--config", config_path,
        "--trainer.accelerator=cpu"
    ]
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        
        process.wait()
    
    print(f"\n{model_name.upper()} training completed with exit code: {process.returncode}")
    return process.returncode == 0

def evaluate_model(model_name):
    """Evaluate a model and extract metrics"""
    print(f"\nEvaluating {model_name.upper()}...")
    
    result = subprocess.run(
        [sys.executable, "evaluate_model.py"],
        capture_output=True,
        text=True
    )
    
    # Extract metrics from output
    metrics = {}
    for line in result.stdout.split("\n"):
        if "ID R1:" in line:
            try:
                metrics["ID_R1"] = float(line.split("ID R1:")[1].strip())
            except:
                pass
        elif "ID R2:" in line:
            try:
                metrics["ID_R2"] = float(line.split("ID R2:")[1].strip())
            except:
                pass
        elif "OOD R1:" in line:
            try:
                metrics["OOD_R1"] = float(line.split("OOD R1:")[1].strip())
            except:
                pass
        elif "OOD R2:" in line:
            try:
                metrics["OOD_R2"] = float(line.split("OOD R2:")[1].strip())
            except:
                pass
    
    return metrics, result.stdout

def main():
    print("=" * 70)
    print("Training Failed Models: Mamba, xLSTM, Linear")
    print("=" * 70)
    print(f"\nModels to train: {', '.join([m[0] for m in MODELS])}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will take approximately 1-3 hours depending on your CPU.")
    print("=" * 70)
    
    results = {}
    
    for model_name, config_path in MODELS:
        if not os.path.exists(config_path):
            print(f"\n⚠ Config not found: {config_path}, skipping {model_name}")
            continue
        
        # Train model
        success = train_model(model_name, config_path)
        
        if success:
            # Wait a bit for checkpoint to be saved
            time.sleep(2)
            
            # Evaluate
            metrics, output = evaluate_model(model_name)
            results[model_name] = metrics
            
            # Save evaluation output
            with open(f"evaluation_{model_name}.txt", "w") as f:
                f.write(output)
            
            print(f"\n✓ {model_name.upper()} completed")
            if metrics:
                print(f"  ID R1: {metrics.get('ID_R1', 'N/A')}")
                print(f"  ID R2: {metrics.get('ID_R2', 'N/A')}")
        else:
            print(f"\n✗ {model_name.upper()} training failed")
            results[model_name] = None
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<15} {'ID R1':<10} {'ID R2':<10} {'OOD R1':<10} {'OOD R2':<10}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        if metrics:
            print(f"{model_name:<15} "
                  f"{metrics.get('ID_R1', 0):<10.3f} "
                  f"{metrics.get('ID_R2', 0):<10.3f} "
                  f"{metrics.get('OOD_R1', 0):<10.3f} "
                  f"{metrics.get('OOD_R2', 0):<10.3f}")
        else:
            print(f"{model_name:<15} {'FAILED':<10}")
    
    print("=" * 70)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

