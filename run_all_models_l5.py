#!/usr/bin/env python3
"""
Script to train and test all models (Transformer, Linear, LSTM, Mamba, xLSTM) 
for aNbNcN grammar (L5) and display results in a combined table format.
"""

import sys
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import time

from rule_extrapolation.runner import LightningGrammarModule
from rule_extrapolation.datamodule import GrammarDataModule


def convert_to_float(value):
    """Convert tensor or other types to float."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def train_and_evaluate_model(model_name, config_base, datamodule_config):
    """Train and evaluate a single model."""
    print(f"\n{'='*80}")
    print(f"Training {model_name.upper()} model...")
    print(f"{'='*80}")

    start_time = time.time()

    # Create model-specific config
    config = OmegaConf.create(config_base.copy())
    config.model.model = model_name.lower()

    # Set model-specific parameters
    if model_name.lower() == "linear":
        config.model.bias = True
        if "dim_model" not in config.model:
            config.model.dim_model = 10
    elif model_name.lower() == "lstm":
        config.model.hidden_dim = 64
        config.model.num_layers = 5
        config.model.dropout = 0.4
        config.model.embedding_dim = 16
        if "embedding_dim" in config.model:
            config.model.dim_model = config.model.embedding_dim
    elif model_name.lower() == "mamba":
        config.model.n_layers = 10
        config.model.d_state = 16
        config.model.d_conv = 8
        config.model.d_model = 32
    elif model_name.lower() == "xlstm":
        config.model.num_blocks = 6
        config.model.xlstm_embedding_dim = 64
        config.model.slstm_at = [1]
    elif model_name.lower() == "transformer":
        pass

    # Create datamodule
    datamodule = GrammarDataModule(
        num_train=datamodule_config["num_train"],
        num_val=datamodule_config["num_val"],
        num_test=datamodule_config["num_test"],
        max_length=datamodule_config["max_length"],
        batch_size=datamodule_config["batch_size"],
        grammar=datamodule_config["grammar"],
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Create model with appropriate parameters
    model_params = {
        "num_tokens": config.model.num_tokens,
        "test_prompt_length": config.model.test_prompt_length,
        "max_pred_length": config.model.max_pred_length,
        "lr": config.model.lr,
        "model": config.model.model,
        "grammar": datamodule_config["grammar"],
        "max_data_length": datamodule_config["max_length"],
        "batch_size": datamodule_config["batch_size"],
    }

    # Add model-specific parameters
    if model_name.lower() == "transformer":
        model_params.update(
            {
                "dim_model": config.model.dim_model,
                "dim_feedforward": config.model.dim_feedforward,
                "num_heads": config.model.num_heads,
                "num_decoder_layers": config.model.num_decoder_layers,
                "dropout_p": config.model.dropout_p,
                "layer_norm_eps": config.model.layer_norm_eps,
            }
        )
    elif model_name.lower() == "linear":
        # Linear uses dim_model as embedding_dim
        model_params.update(
            {
                "bias": config.model.get("bias", True),
                "dim_model": config.model.dim_model,  # Used as embedding_dim
            }
        )
    elif model_name.lower() == "lstm":
        # LSTM uses dim_model as embedding_dim
        model_params.update(
            {
                "dim_model": config.model.get(
                    "embedding_dim", config.model.dim_model
                ),  # Used as embedding_dim
                "hidden_dim": config.model.hidden_dim,
                "num_layers": config.model.num_layers,
                "dropout": config.model.dropout,
            }
        )
    elif model_name.lower() == "mamba":
        model_params.update(
            {
                "n_layers": config.model.n_layers,
                "d_state": config.model.d_state,
                "d_conv": config.model.d_conv,
                "d_model": config.model.d_model,
            }
        )
    elif model_name.lower() == "xlstm":
        model_params.update(
            {
                "num_blocks": config.model.num_blocks,
                "xlstm_embedding_dim": config.model.xlstm_embedding_dim,
                "slstm_at": config.model.slstm_at,
            }
        )
        # xLSTM can now work on CPU (runner.py has been updated)

    try:
        model = LightningGrammarModule(**model_params)
    except Exception as e:
        print(f"Error creating {model_name} model: {e}")
        if "CUDA" in str(e) or "cuda" in str(e).lower():
            print(f"  Note: {model_name} may require CUDA. Skipping...")
        return None

    # Setup trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="Val/loss",
        mode="min",
        save_top_k=1,
        filename=f"{model_name}_l5_fast-{{epoch:02d}}-{{Val/loss:.4f}}",
        verbose=False,
    )

    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_progress_bar=False,  # Disable to reduce output
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=False,
        benchmark=True,
    )

    # Train
    print(f"Training {model_name}...")
    try:
        trainer.fit(model, datamodule=datamodule)
        model = trainer.model
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        return None

    # Setup test datamodule
    datamodule.setup("test")

    # Evaluate
    print(f"Evaluating {model_name}...")
    model.eval()
    model.freeze()

    test_dl = datamodule.test_dataloader()

    # Calculate test loss
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in test_dl:
            try:
                _, _, _, loss = model._forward(batch)
                total_loss += convert_to_float(loss)
                num_batches += 1
                if num_batches >= 10:  # Limit for speed
                    break
            except Exception as e:
                print(f"Error calculating loss for {model_name}: {e}")
                break
    test_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    # Evaluate prompt predictions
    try:
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

        id_r1 = convert_to_float(metrics.rule_1_accuracy)
        id_r2 = convert_to_float(metrics.rule_2_accuracy)
        ood_r1 = convert_to_float(ood_metrics.rule_1_accuracy)
        ood_r2_completion = convert_to_float(ood_metrics.rule_2_completion_accuracy)
    except Exception as e:
        print(f"Error evaluating prompts for {model_name}: {e}")
        id_r1 = id_r2 = ood_r1 = ood_r2_completion = 0.0

    elapsed_time = time.time() - start_time
    print(f"{model_name} completed in {elapsed_time:.1f} seconds")

    return {
        "model": model_name,
        "test_loss": test_loss,
        "id_r1": id_r1,
        "id_r2": id_r2,
        "ood_r1": ood_r1,
        "ood_r2_completion": ood_r2_completion,
        "time": elapsed_time,
    }


def print_combined_results_table(results_list):
    """Print combined results table for all models."""
    print("\n" + "=" * 100)
    print(
        "Table 6: Test loss and rule-following accuracies for the context-sensitive language L5 = {a^n b^n c^n}: the Transformer can extrapolate (R1) the best"
    )
    print("=" * 100)
    print(
        f"{'Model':<15} {'Test loss':<20} {'ID R1':<15} {'ID R2':<15} {'OOD R1':<15} {'OOD R2 completion':<20}"
    )
    print("-" * 100)

    # Model order as in the image - show only models that have results
    model_order = ["Transformer", "Linear", "LSTM", "Mamba", "xLSTM"]

    # Filter to only show models with results
    available_models = [r["model"] for r in results_list if r is not None]

    for model_name in model_order:
        # Find results for this model
        model_results = None
        for r in results_list:
            if r and r["model"].lower() == model_name.lower():
                model_results = r
                break

        # Skip models that weren't run (only show available ones)
        if model_results is None and model_name not in available_models:
            continue

        if model_results is None:
            print(
                f"{model_name:<15} {'N/A':<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<20}"
            )
            continue

        # Round values
        test_loss = round(model_results["test_loss"], 3)
        id_r1 = round(model_results["id_r1"], 3)
        id_r2 = round(model_results["id_r2"], 3)
        ood_r1 = round(model_results["ood_r1"], 3)
        ood_r2 = round(model_results["ood_r2_completion"], 3)

        # Format test loss
        if test_loss == float("inf") or test_loss == float("-inf"):
            loss_str = "N/A"
        else:
            loss_str = f"{test_loss:.3f}"

        print(
            f"{model_name:<15} {loss_str:<20} {id_r1:.3f}            {id_r2:.3f}            {ood_r1:.3f}            {ood_r2:.3f}"
        )

    print("=" * 100)
    print()


def main():
    # Base config with increased values for better accuracy
    base_config = {
        "seed_everything": 42,
        "trainer": {
            "logger": False,
            "accelerator": "auto",
            "max_epochs": 1000,
            "limit_train_batches": None,
            "limit_val_batches": None,
            "check_val_every_n_epoch": 50,
            "num_sanity_val_steps": 0,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "deterministic": False,
            "benchmark": True,
        },
        "model": {
            "num_tokens": 6,
            "dim_model": 10,
            "dim_feedforward": 1024,
            "num_heads": 5,
            "test_prompt_length": 8,
            "max_pred_length": 300,
            "num_decoder_layers": 7,
            "dropout_p": 0.1,
            "lr": 0.002,
            "layer_norm_eps": 6e-3,
            "model": "transformer",
            "embedding_dim": 16,
            "hidden_dim": 64,
            "num_layers": 5,
            "dropout": 0.4,
            "bias": True,
            "n_layers": 10,
            "d_state": 16,
            "d_conv": 8,
            "d_model": 32,
            "num_blocks": 6,
            "xlstm_embedding_dim": 64,
            "slstm_at": [1],
        },
        "data": {
            "num_train": 512,
            "num_val": 256,
            "num_test": 256,
            "max_length": 256,
            "batch_size": 128,
            "grammar": "aNbNcN",
        },
    }

    datamodule_config = base_config["data"]

    # Models to run - include all models
    models = ["Transformer", "Linear", "LSTM"]

    # Try to add Mamba if available
    mamba_available = False
    try:
        from rule_extrapolation.runner import MambaLM

        if MambaLM is not None:
            mamba_available = True
            models.append("Mamba")
            print("✓ Mamba module found, will run Mamba model")
    except Exception as e:
        print(f"✗ Mamba module not available: {e}")
        print("  To install Mamba:")
        print(
            "    1. Initialize git submodule: git submodule update --init --recursive"
        )
        print(
            "    2. Or manually clone: cd mamba && git clone https://github.com/rpatrik96/mamba.py ."
        )
        print("  Mamba will be skipped")

    # Add xLSTM (works on CPU with mLSTM-only configuration)
    models.append("xLSTM")
    if torch.cuda.is_available():
        print("✓ CUDA available, will run xLSTM with full sLSTM blocks")
    else:
        print(
            "✓ xLSTM will run on CPU with mLSTM-only configuration (sLSTM requires CUDA)"
        )

    # Store results
    all_results = []
    total_start_time = time.time()

    # Run each model
    for model_name in models:
        try:
            result = train_and_evaluate_model(
                model_name, base_config, datamodule_config
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Failed to run {model_name}: {e}")
            all_results.append(None)

    total_time = time.time() - total_start_time

    # Print combined results table
    print_combined_results_table(all_results)

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("\nIndividual model times:")
    for result in all_results:
        if result:
            print(f"  {result['model']}: {result['time']:.1f} seconds")
    print("=" * 100)


if __name__ == "__main__":
    main()
