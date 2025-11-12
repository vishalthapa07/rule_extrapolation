#!/usr/bin/env python3
"""
Train all models (Transformer, LSTM, Mamba, xLSTM, Linear) for L5 grammar
L5 = parentheses_and_brackets (Dyck language with both parentheses and brackets)
"""

import subprocess
import sys
import os
import time
import glob
from datetime import datetime

MODELS = [
    ("transformer", "configs/L5_all_models_transformer.yaml"),
    ("lstm", "configs/L5_all_models_lstm.yaml"),
    ("mamba", "configs/L5_all_models_mamba.yaml"),
    ("xlstm", "configs/L5_all_models_xlstm.yaml"),
    ("linear", "configs/L5_all_models_linear.yaml"),
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
    print(f"Training {model_name.upper()} Model for L5 Grammar")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    
    log_file = f"training_L5_{model_name}.log"
    
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

def evaluate_model(model_name, config_path):
    """Evaluate a model and extract metrics"""
    print(f"\nEvaluating {model_name.upper()}...")
    
    # Create a temporary evaluation script that uses the correct config
    eval_script = f"""
import sys
import os
import glob
sys.path.insert(0, os.getcwd())

from rule_extrapolation.runner import LightningGrammarModule
from rule_extrapolation.datamodule import GrammarDataModule
from omegaconf import OmegaConf

# Find latest checkpoint
version_dirs = glob.glob("lightning_logs/version_*")
if version_dirs:
    latest_version = max(version_dirs, key=os.path.getmtime)
    checkpoint_dir = os.path.join(latest_version, "checkpoints")
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        if checkpoints:
            checkpoint_path = max(checkpoints, key=os.path.getmtime)
            
            # Load config
            config = OmegaConf.load("{config_path}")
            
            # Load model
            model = LightningGrammarModule.load_from_checkpoint(checkpoint_path)
            model.eval()
            model.to("cpu")
            
            # Setup datamodule
            datamodule = GrammarDataModule(
                num_train=config.data.num_train,
                num_val=config.data.num_val,
                num_test=config.data.num_test,
                max_length=config.data.max_length,
                batch_size=config.data.batch_size,
                grammar=config.data.grammar,
            )
            datamodule.setup("validate")
            
            # Evaluate
            (
                prompts, metrics, prompts_finished, metrics_finished,
                ood_prompts, ood_metrics, ood_prompts_finished, ood_metrics_finished,
                sos_prompts, sos_metrics, sos_prompts_finished, sos_metrics_finished,
            ) = model.eval_prompt_prediction()
            
            print("=" * 60)
            print("RESULTS - ID Metrics (In-Distribution)")
            print("=" * 60)
            print(f"ID R1 Accuracy: {{metrics.rule_1_accuracy:.4f}}")
            print(f"ID R2 Accuracy: {{metrics.rule_2_accuracy:.4f}}")
            print(f"Grammatical Accuracy: {{metrics.grammatical_accuracy:.4f}}")
            
            print("\\n" + "=" * 60)
            print("RESULTS - OOD Metrics (Out-of-Distribution)")
            print("=" * 60)
            print(f"OOD R1 Accuracy: {{ood_metrics.rule_1_accuracy:.4f}}")
            print(f"OOD R2 Accuracy: {{ood_metrics.rule_2_accuracy:.4f}}")
            
            print("\\n" + "=" * 60)
            print("Summary Table Format")
            print("=" * 60)
            print(f"ID R1:  {{metrics.rule_1_accuracy:.3f}}")
            print(f"ID R2:  {{metrics.rule_2_accuracy:.3f}}")
            print(f"OOD R1: {{ood_metrics.rule_1_accuracy:.3f}}")
            print(f"OOD R2: {{ood_metrics.rule_2_accuracy:.3f}}")
"""
    
    result = subprocess.run(
        [sys.executable, "-c", eval_script],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
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
    print("Training All Models for L5 Grammar (parentheses_and_brackets)")
    print("=" * 70)
    print(f"\nModels to train: {', '.join([m[0] for m in MODELS])}")
    print(f"Total models: {len(MODELS)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nL5 Grammar: Dyck language with parentheses () and brackets []")
    print("Rule 1 (R1): Matched brackets")
    print("Rule 2 (R2): Matched parentheses")
    print("\nThis will take approximately 2-5 hours depending on your CPU.")
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
            metrics, output = evaluate_model(model_name, config_path)
            results[model_name] = metrics
            
            # Save evaluation output
            with open(f"evaluation_L5_{model_name}.txt", "w") as f:
                f.write(output)
            
            print(f"\n✓ {model_name.upper()} completed")
            if metrics:
                print(f"  ID R1: {metrics.get('ID_R1', 'N/A')}")
                print(f"  ID R2: {metrics.get('ID_R2', 'N/A')}")
        else:
            print(f"\n✗ {model_name.upper()} training failed")
            results[model_name] = None
    
    # Print summary table
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY - L5 Grammar (parentheses_and_brackets)")
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
    
    # Save results to file
    with open("L5_all_models_results.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ALL MODELS TRAINING RESULTS - L5 Grammar\n")
        f.write("Grammar: parentheses_and_brackets (Dyck language)\n")
        f.write("Rule 1 (R1): Matched brackets\n")
        f.write("Rule 2 (R2): Matched parentheses\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Model':<15} {'ID R1':<10} {'ID R2':<10} {'OOD R1':<10} {'OOD R2':<10}\n")
        f.write("-" * 70 + "\n")
        for model_name, metrics in results.items():
            if metrics:
                f.write(f"{model_name:<15} "
                       f"{metrics.get('ID_R1', 0):<10.3f} "
                       f"{metrics.get('ID_R2', 0):<10.3f} "
                       f"{metrics.get('OOD_R1', 0):<10.3f} "
                       f"{metrics.get('OOD_R2', 0):<10.3f}\n")
            else:
                f.write(f"{model_name:<15} {'FAILED':<10}\n")
    
    print("\nResults saved to: L5_all_models_results.txt")
    
    # Also create a LaTeX table format
    with open("L5_all_models_results_latex.txt", "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Model & ID R1 & ID R2 & OOD R1 & OOD R2 \\\\\n")
        f.write("\\midrule\n")
        for model_name, metrics in results.items():
            if metrics:
                model_display = model_name.capitalize() if model_name != "lstm" else "LSTM"
                f.write(f"{model_display} & "
                       f"{metrics.get('ID_R1', 0):.3f} & "
                       f"{metrics.get('ID_R2', 0):.3f} & "
                       f"{metrics.get('OOD_R1', 0):.3f} & "
                       f"{metrics.get('OOD_R2', 0):.3f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{L5 Grammar (parentheses\\_and\\_brackets) Results}\n")
        f.write("\\end{table}\n")
    
    print("LaTeX table saved to: L5_all_models_results_latex.txt")

if __name__ == "__main__":
    main()

