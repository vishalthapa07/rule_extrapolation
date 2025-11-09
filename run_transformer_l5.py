#!/usr/bin/env python3
"""
Script to train and test transformer model for aNbNcN grammar (L5)
and display results in a table format similar to the paper.
"""

import sys
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from rule_extrapolation.runner import LightningGrammarModule
from rule_extrapolation.datamodule import GrammarDataModule


def print_results_table(test_loss, id_r1, id_r2, ood_r1, ood_r2_completion):
    """Print results in a table format similar to the image."""
    print("\n" + "=" * 100)
    print(
        "Table 6: Test loss and rule-following accuracies for the context-sensitive language L5 = {a^n b^n c^n}"
    )
    print("=" * 100)
    print(
        f"{'Model':<15} {'Test loss':<20} {'ID R1':<15} {'ID R2':<15} {'OOD R1':<15} {'OOD R2 completion':<20}"
    )
    print("-" * 100)

    # Convert to float if tensor, then round to 3 decimal places
    if isinstance(test_loss, (list, tuple)):
        loss_val = float(
            test_loss[0].item() if hasattr(test_loss[0], "item") else test_loss[0]
        )
        std_val = float(
            test_loss[1].item() if hasattr(test_loss[1], "item") else test_loss[1]
        )
        loss_rounded = round(loss_val, 3)
        std_rounded = round(std_val, 3)
        loss_str = f"{loss_rounded:.3f} ± {std_rounded:.3f}"
    else:
        loss_val = float(test_loss.item() if hasattr(test_loss, "item") else test_loss)
        loss_rounded = round(loss_val, 3)
        loss_str = f"{loss_rounded:.3f}"

    # Convert to float if tensor, then round accuracies to 3 decimal places
    id_r1_val = float(id_r1.item() if hasattr(id_r1, "item") else id_r1)
    id_r2_val = float(id_r2.item() if hasattr(id_r2, "item") else id_r2)
    ood_r1_val = float(ood_r1.item() if hasattr(ood_r1, "item") else ood_r1)
    ood_r2_val = float(
        ood_r2_completion.item()
        if hasattr(ood_r2_completion, "item")
        else ood_r2_completion
    )

    id_r1_rounded = round(id_r1_val, 3)
    id_r2_rounded = round(id_r2_val, 3)
    ood_r1_rounded = round(ood_r1_val, 3)
    ood_r2_rounded = round(ood_r2_val, 3)

    id_r1_str = f"{id_r1_rounded:.3f}"
    id_r2_str = f"{id_r2_rounded:.3f}"
    ood_r1_str = f"{ood_r1_rounded:.3f}"
    ood_r2_str = f"{ood_r2_rounded:.3f}"

    print(
        f"{'Transformer':<15} {loss_str:<20} {id_r1_str:<15} {id_r2_str:<15} {ood_r1_str:<15} {ood_r2_str:<20}"
    )
    print("=" * 100)
    print()

    # Also return rounded values for summary
    return {
        "test_loss": loss_rounded
        if not isinstance(test_loss, (list, tuple))
        else (loss_rounded, std_rounded),
        "id_r1": id_r1_rounded,
        "id_r2": id_r2_rounded,
        "ood_r1": ood_r1_rounded,
        "ood_r2_completion": ood_r2_rounded,
    }


def run_training_and_testing(config_path):
    """Run training and testing, then extract and display results."""

    print("Loading configuration...")
    config = OmegaConf.load(config_path)

    # Create datamodule
    print("Preparing data...")
    datamodule = GrammarDataModule(
        num_train=config.data.num_train,
        num_val=config.data.num_val,
        num_test=config.data.num_test,
        max_length=config.data.max_length,
        batch_size=config.data.batch_size,
        grammar=config.data.grammar,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Create model
    print("Creating model...")
    model = LightningGrammarModule(
        num_tokens=config.model.num_tokens,
        dim_model=config.model.dim_model,
        dim_feedforward=config.model.dim_feedforward,
        num_heads=config.model.num_heads,
        test_prompt_length=config.model.test_prompt_length,
        max_pred_length=config.model.max_pred_length,
        num_decoder_layers=config.model.num_decoder_layers,
        dropout_p=config.model.dropout_p,
        lr=config.model.lr,
        layer_norm_eps=config.model.layer_norm_eps,
        model=config.model.model,
        grammar=config.data.grammar,
        max_data_length=config.data.max_length,
        batch_size=config.data.batch_size,
    )

    # Setup trainer for training
    print("Setting up trainer for training...")
    checkpoint_callback = ModelCheckpoint(
        monitor="Val/loss",
        mode="min",
        save_top_k=1,
        filename="transformer_l5_fast-{epoch:02d}-{Val/loss:.4f}",
        verbose=False,
    )

    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
        logger=False,  # Disable logging for speed
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=False,
        benchmark=True,
    )

    # Train
    print("Starting training...")
    print(
        f"Training for {config.trainer.max_epochs} epochs with {config.trainer.limit_train_batches} batches per epoch..."
    )
    trainer.fit(model, datamodule=datamodule)

    print(f"\nTraining completed!")

    # Use the trained model (it's already the best after training)
    # The trainer keeps the best model in memory
    model = trainer.model

    # Setup test datamodule
    datamodule.setup("test")

    # Manual evaluation on test data
    print("Running evaluation on test data...")
    model.eval()
    model.freeze()

    # Get test dataloader
    test_dl = datamodule.test_dataloader()

    # Calculate test loss
    print("Calculating test loss...")
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in test_dl:
            _, _, _, loss = model._forward(batch)
            total_loss += loss.item()
            num_batches += 1
            if num_batches >= 10:  # Limit for speed
                break
    test_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Evaluate prompt predictions for rule-following accuracies
    print("Evaluating rule-following accuracies...")
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

    id_r1 = metrics.rule_1_accuracy
    id_r2 = metrics.rule_2_accuracy
    ood_r1 = ood_metrics.rule_1_accuracy
    ood_r2_completion = ood_metrics.rule_2_completion_accuracy

    print(f"\nEvaluation results (raw):")
    print(f"  Test loss: {test_loss:.6f}")
    print(f"  ID R1: {id_r1:.6f}")
    print(f"  ID R2: {id_r2:.6f}")
    print(f"  OOD R1: {ood_r1:.6f}")
    print(f"  OOD R2 completion: {ood_r2_completion:.6f}")

    # Display results table with rounded values
    rounded_results = print_results_table(
        test_loss, id_r1, id_r2, ood_r1, ood_r2_completion
    )

    return rounded_results


if __name__ == "__main__":
    config_path = "configs/fast_transformer_l5.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    results = run_training_and_testing(config_path)
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY (Rounded to 3 decimal places):")
    print("=" * 100)
    for key, value in results.items():
        if isinstance(value, tuple):
            print(f"  {key}: {value[0]:.3f} ± {value[1]:.3f}")
        else:
            print(f"  {key}: {value:.3f}")
    print("=" * 100)
