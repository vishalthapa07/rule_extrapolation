#!/usr/bin/env python3
"""
Script to evaluate a trained model and extract ID R1 and ID R2 metrics
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from rule_extrapolation.runner import LightningGrammarModule
from rule_extrapolation.datamodule import GrammarDataModule
import sys

def evaluate_model(checkpoint_path, config_path=None):
    """
    Load a trained model and evaluate it to get ID R1 and ID R2 metrics
    """
    print("=" * 60)
    print("Evaluating Model for ID R1 and ID R2 Metrics")
    print("=" * 60)
    
    # Load model from checkpoint
    print(f"\nLoading model from: {checkpoint_path}")
    model = LightningGrammarModule.load_from_checkpoint(checkpoint_path)
    
    # Create datamodule
    if config_path:
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)
        datamodule = GrammarDataModule(
            num_train=config.data.num_train,
            num_val=config.data.num_val,
            num_test=config.data.num_test,
            max_length=config.data.max_length,
            batch_size=config.data.batch_size,
            grammar=config.data.grammar,
        )
    else:
        # Use default values based on the checkpoint
        datamodule = GrammarDataModule(
            num_train=256,
            num_val=128,
            num_test=128,
            max_length=64,
            batch_size=64,
            grammar="aNbN",
        )
    
    datamodule.setup("validate")
    
    # Set model to eval mode
    model.eval()
    model.to("cpu")
    
    # Now manually evaluate to get the metrics
    print("\n" + "=" * 60)
    print("Computing ID R1 and ID R2 metrics...")
    print("=" * 60)
    
    # Evaluate on test prompts
    (
        prompts,
        metrics,
        prompts_finished,
        metrics_finished,
        ood_prompts,
        ood_metrics,
        ood_prompts_finished,
        ood_metrics_finished,
        sos_prompts,
        sos_metrics,
        sos_prompts_finished,
        sos_metrics_finished,
    ) = model.eval_prompt_prediction()
    
    print("\n" + "=" * 60)
    print("RESULTS - ID Metrics (In-Distribution)")
    print("=" * 60)
    print(f"ID R1 Accuracy (same number of a's and b's): {metrics.rule_1_accuracy:.4f}")
    print(f"ID R2 Accuracy (a's before b's): {metrics.rule_2_accuracy:.4f}")
    print(f"Grammatical Accuracy: {metrics.grammatical_accuracy:.4f}")
    print(f"Finished Accuracy: {metrics.finished_accuracy:.4f}")
    
    print("\n" + "=" * 60)
    print("RESULTS - OOD Metrics (Out-of-Distribution)")
    print("=" * 60)
    print(f"OOD R1 Accuracy: {ood_metrics.rule_1_accuracy:.4f}")
    print(f"OOD R2 Accuracy: {ood_metrics.rule_2_accuracy:.4f}")
    print(f"OOD Grammatical Accuracy: {ood_metrics.grammatical_accuracy:.4f}")
    print(f"OOD Finished Accuracy: {ood_metrics.finished_accuracy:.4f}")
    
    print("\n" + "=" * 60)
    print("Summary Table Format (like Table 4)")
    print("=" * 60)
    print(f"ID R1:  {metrics.rule_1_accuracy:.3f}")
    print(f"ID R2:  {metrics.rule_2_accuracy:.3f}")
    print(f"OOD R1: {ood_metrics.rule_1_accuracy:.3f}")
    print(f"OOD R2: {ood_metrics.rule_2_accuracy:.3f}")
    print("=" * 60)
    
    return metrics, ood_metrics

if __name__ == "__main__":
    import os
    
    # Find the latest checkpoint
    import glob
    version_dirs = glob.glob("lightning_logs/version_*")
    if version_dirs:
        latest_version = max(version_dirs, key=os.path.getmtime)
        checkpoint_dir = os.path.join(latest_version, "checkpoints")
        print(f"Using latest version directory: {latest_version}")
    else:
        checkpoint_dir = "lightning_logs/version_47/checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        print("Please provide a checkpoint path as argument.")
        sys.exit(1)
    
    # Find the best checkpoint (lowest validation loss)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        sys.exit(1)
    
    # Use the last checkpoint (usually the best one)
    checkpoint_file = sorted(checkpoints)[-1]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    
    print(f"Using checkpoint: {checkpoint_path}\n")
    
    # Evaluate
    config_path = "configs/config_balanced.yaml"
    if not os.path.exists(config_path):
        config_path = "configs/config_fast.yaml"
    
    metrics, ood_metrics = evaluate_model(
        checkpoint_path, 
        config_path=config_path
    )

