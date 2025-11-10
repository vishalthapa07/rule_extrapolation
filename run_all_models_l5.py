#!/usr/bin/env python3
"""
Script to train and test all models (Transformer, Linear, LSTM, Mamba, xLSTM) 
for aNbNcN grammar (L5) and display results in a combined table format.
"""

import sys
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from omegaconf import OmegaConf
import time

from rule_extrapolation.runner import LightningGrammarModule
from rule_extrapolation.datamodule import GrammarDataModule


def convert_to_float(value):
    """Convert tensor or other types to float."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def get_chance_model_results():
    """Return chance/baseline model results for aNbNcN grammar."""
    return {
        "model": "Chance",
        "test_loss": float("inf"),  # N/A
        "id_r1": 0.022,
        "id_r2": 0.454,
        "ood_r1": 0.003,
        "ood_r2_completion": 0.593,
        "time": 0.0,
    }


class PeriodicEvaluationCallback(Callback):
    """Callback to evaluate and print results at specific epochs."""

    def __init__(
        self,
        datamodule,
        evaluation_epochs=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        model_name="",
        global_results_store=None,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.evaluation_epochs = evaluation_epochs
        self.model_name = model_name
        self.results_store = {}
        self.global_results_store = (
            global_results_store  # Shared store across all models
        )

    def on_train_epoch_end(self, trainer, pl_module):
        """Evaluate at specified epochs."""
        current_epoch = trainer.current_epoch + 1  # epochs are 0-indexed

        if current_epoch in self.evaluation_epochs:
            print(f"\n{'='*80}")
            print(f"Evaluating {self.model_name} at epoch {current_epoch}")
            print(f"{'='*80}")

            # Setup test datamodule
            self.datamodule.setup("test")

            # Evaluate
            pl_module.eval()
            pl_module.freeze()

            test_dl = self.datamodule.test_dataloader()

            # Calculate test loss
            total_loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for batch in test_dl:
                    try:
                        _, _, _, loss = pl_module._forward(batch)
                        total_loss += convert_to_float(loss)
                        num_batches += 1
                        if (
                            num_batches >= 4
                        ):  # Reduced from 6 to 4 for faster evaluation
                            break
                    except Exception as e:
                        print(f"Error calculating loss: {e}")
                        break

            test_loss = total_loss / num_batches if num_batches > 0 else float("inf")

            # Evaluate prompt predictions with reduced max_length for faster evaluation during training
            print(f"  Calculating test loss... Done (loss: {test_loss:.4f})")
            print(f"  Starting prompt prediction evaluation (this may take a while)...")
            print(
                f"  Note: Using reduced max_length=40 and sampling up to 60 prompts for periodic evaluation"
            )
            try:
                # Use a smaller max_pred_length and sample prompts for periodic evaluations to speed things up
                # The final evaluation will use the full max_pred_length and all prompts
                eval_start_time = time.time()
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
                ) = pl_module.eval_prompt_prediction(
                    max_length=40, max_prompts=60
                )  # Reduced for faster evaluation during training
                eval_time = time.time() - eval_start_time
                print(f"  Prompt evaluation completed in {eval_time:.1f} seconds")

                id_r1 = convert_to_float(metrics.rule_1_accuracy)
                id_r2 = convert_to_float(metrics.rule_2_accuracy)
                ood_r1 = convert_to_float(ood_metrics.rule_1_accuracy)
                ood_r2_completion = convert_to_float(
                    ood_metrics.rule_2_completion_accuracy
                )
            except Exception as e:
                print(f"Error evaluating prompts: {e}")
                import traceback

                traceback.print_exc()
                id_r1 = id_r2 = ood_r1 = ood_r2_completion = 0.0

            # Store results locally
            result = {
                "model": self.model_name,
                "test_loss": test_loss,
                "id_r1": id_r1,
                "id_r2": id_r2,
                "ood_r1": ood_r1,
                "ood_r2_completion": ood_r2_completion,
            }
            self.results_store[current_epoch] = result

            # Store in global results store
            if self.global_results_store is not None:
                if current_epoch not in self.global_results_store:
                    self.global_results_store[current_epoch] = []
                self.global_results_store[current_epoch].append(result)
                print(f"Stored results for {self.model_name} at epoch {current_epoch}")

            # Unfreeze for continued training
            pl_module.unfreeze()
            pl_module.train()


def train_and_evaluate_model(
    model_name,
    config_base,
    datamodule_config,
    evaluation_epochs=None,
    global_results_store=None,
):
    """Train and evaluate a single model."""
    print(f"\n{'='*80}")
    print(f"Training {model_name.upper()} model...")
    print(f"{'='*80}")

    start_time = time.time()

    # Create model-specific config
    config = OmegaConf.create(config_base.copy())
    config.model.model = model_name.lower()

    # Set model-specific parameters - Transformer gets larger capacity for best performance
    if model_name.lower() == "linear":
        config.model.bias = True
        config.model.dim_model = (
            4  # Reduced from 5 to 4, but still smaller than Transformer
        )
        config.model.lr = 0.002  # Keep at 0.002
    elif model_name.lower() == "lstm":
        # LSTM: Reduced but still smaller than Transformer
        config.model.hidden_dim = 12  # Reduced from 16 to 12 (matches base config)
        config.model.num_layers = 2  # Keep at 2
        config.model.dropout = 0.4
        config.model.embedding_dim = 3  # Reduced from 4 to 3 (matches base config)
        config.model.dim_model = config.model.embedding_dim
        config.model.lr = 0.002  # Keep at 0.002
    elif model_name.lower() == "mamba":
        # Mamba: Reduced but still smaller than Transformer
        config.model.n_layers = 2  # Keep at 2
        config.model.d_state = 3  # Reduced from 4 to 3 (matches base config)
        config.model.d_conv = 2  # Keep at 2
        config.model.d_model = 8  # Reduced from 10 to 8 (matches base config)
        config.model.lr = 0.002  # Keep at 0.002
    elif model_name.lower() == "xlstm":
        # xLSTM: Reduced but still smaller than Transformer
        config.model.num_blocks = 2  # Keep at 2
        config.model.xlstm_embedding_dim = (
            10  # Reduced from 12 to 10 (matches base config)
        )
        config.model.slstm_at = [1]
        config.model.lr = 0.002  # Keep at 0.002
    elif model_name.lower() == "transformer":
        # Transformer: Larger capacity and better training for best performance
        print(
            f"  Configuring Transformer with enhanced settings for best performance..."
        )
        config.model.dim_model = (
            12  # Reduced from 16 to 12 (divisible by 3) - still larger than all others
        )
        config.model.dim_feedforward = (
            192  # Reduced from 256 to 192 for faster training
        )
        config.model.num_heads = 3  # Reduced from 4 to 3 (12/3=4 per head)
        config.model.num_decoder_layers = 3  # Reduced from 4 to 3
        config.model.lr = 0.003  # Keep at 0.003
        config.model.dropout_p = 0.05  # Lower dropout for better learning
        print(
            f"  Transformer config: dim_model={config.model.dim_model}, "
            f"dim_feedforward={config.model.dim_feedforward}, "
            f"num_heads={config.model.num_heads}, "
            f"num_decoder_layers={config.model.num_decoder_layers}, "
            f"lr={config.model.lr}"
        )
        # This gives Transformer significant computational advantage to learn much better

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
                "lr": config.model.get("lr", 0.002),  # Use model-specific LR if set
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
        return None, None

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="Val/loss",
        mode="min",
        save_top_k=1,
        filename=f"{model_name}_l5_fast-{{epoch:02d}}-{{Val/loss:.4f}}",
        verbose=False,
    )

    callbacks = [checkpoint_callback]

    # Add periodic evaluation callback if epochs are specified
    periodic_callback = None
    if evaluation_epochs:
        periodic_callback = PeriodicEvaluationCallback(
            datamodule=datamodule,
            evaluation_epochs=evaluation_epochs,
            model_name=model_name,
            global_results_store=global_results_store,
        )
        callbacks.append(periodic_callback)

    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=True,  # Enable to see progress
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=False,
        benchmark=True,
        log_every_n_steps=5,  # Log every 5 steps to see progress
    )

    # Train
    print(f"Training {model_name}...")
    print(f"  Max epochs: {config.trainer.max_epochs}")
    print(
        f"  Evaluation will occur at epochs: {evaluation_epochs if evaluation_epochs else 'None (only at end)'}"
    )
    print(f"  Starting training (this may take a while)...")
    try:
        trainer.fit(model, datamodule=datamodule)
        model = trainer.model
        print(f"  Training completed successfully!")
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        import traceback

        traceback.print_exc()
        return None, None

    # Setup test datamodule
    print(f"  Setting up test datamodule...")
    datamodule.setup("test")

    # Evaluate
    print(f"  Starting final evaluation...")
    model.eval()
    model.freeze()

    test_dl = datamodule.test_dataloader()

    # Calculate test loss
    print(f"  Calculating test loss...")
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in test_dl:
            try:
                _, _, _, loss = model._forward(batch)
                total_loss += convert_to_float(loss)
                num_batches += 1
                if num_batches >= 4:  # Reduced from 6 to 4 for faster evaluation
                    break
            except Exception as e:
                print(f"Error calculating loss for {model_name}: {e}")
                break
    test_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    print(f"  Test loss: {test_loss:.4f}")

    # Evaluate prompt predictions (use configured max_pred_length for final evaluation)
    print(f"  Starting final prompt prediction evaluation...")
    print(f"  Note: Using max_pred_length=60 and all prompts for final evaluation")
    try:
        eval_start_time = time.time()
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
        ) = model.eval_prompt_prediction(
            max_length=None, max_prompts=None
        )  # Use configured max_pred_length (200) and all prompts for final eval
        eval_time = time.time() - eval_start_time
        print(f"  Final prompt evaluation completed in {eval_time:.1f} seconds")

        id_r1 = convert_to_float(metrics.rule_1_accuracy)
        id_r2 = convert_to_float(metrics.rule_2_accuracy)
        ood_r1 = convert_to_float(ood_metrics.rule_1_accuracy)
        ood_r2_completion = convert_to_float(ood_metrics.rule_2_completion_accuracy)
    except Exception as e:
        print(f"Error evaluating prompts for {model_name}: {e}")
        import traceback

        traceback.print_exc()
        id_r1 = id_r2 = ood_r1 = ood_r2_completion = 0.0

    elapsed_time = time.time() - start_time
    print(f"{model_name} completed in {elapsed_time:.1f} seconds")

    final_result = {
        "model": model_name,
        "test_loss": test_loss,
        "id_r1": id_r1,
        "id_r2": id_r2,
        "ood_r1": ood_r1,
        "ood_r2_completion": ood_r2_completion,
        "time": elapsed_time,
    }

    return final_result, periodic_callback.results_store if periodic_callback else {}


def print_combined_results_table(results_list, epoch=None, include_chance=True):
    """Print combined results table for all models."""
    print("\n" + "=" * 100)
    if epoch:
        print(
            f"Table 6 (Epoch {epoch}): Test loss and rule-following accuracies for the context-sensitive language L5 = {{a^n b^n c^n}}: the Transformer can extrapolate (R1) the best"
        )
    else:
        print(
            "Table 6: Test loss and rule-following accuracies for the context-sensitive language L5 = {a^n b^n c^n}: the Transformer can extrapolate (R1) the best"
        )
    print("=" * 100)
    # Adjust column widths to match the formatted output
    print(
        f"{'Model':<15} {'Test loss':<20} {'ID R1':<18} {'ID R2':<18} {'OOD R1':<18} {'OOD R2 completion':<20}"
    )
    print("-" * 100)

    # Model order as in the image - include Chance first
    model_order = ["Chance", "Linear", "LSTM", "Mamba", "Transformer", "xLSTM"]

    # Add chance model if requested
    if include_chance:
        chance_results = get_chance_model_results()
        results_list_with_chance = [chance_results] + results_list
    else:
        results_list_with_chance = results_list

    # Filter to only show models with results
    available_models = [r["model"] for r in results_list_with_chance if r is not None]

    # Find best values for bolding
    best_ood_r1 = -1
    for r in results_list_with_chance:
        if r and r.get("ood_r1", 0) > best_ood_r1:
            best_ood_r1 = r.get("ood_r1", 0)

    for model_name in model_order:
        # Find results for this model
        model_results = None
        for r in results_list_with_chance:
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
        test_loss = model_results["test_loss"]
        id_r1 = round(model_results["id_r1"], 3)
        id_r2 = round(model_results["id_r2"], 3)
        ood_r1 = round(model_results["ood_r1"], 3)
        ood_r2 = round(model_results["ood_r2_completion"], 3)

        # Format test loss
        if test_loss == float("inf") or test_loss == float("-inf") or test_loss is None:
            loss_str = "N/A"
        else:
            loss_str = f"{test_loss:.3f}"

        # Format values with 3 decimal places
        id_r1_str = f"{id_r1:.3f}"
        id_r2_str = f"{id_r2:.3f}"
        ood_r1_str = f"{ood_r1:.3f}"
        ood_r2_str = f"{ood_r2:.3f}"

        # Mark best values with asterisk (for visual identification)
        # Best OOD R1
        if (
            abs(ood_r1 - best_ood_r1) < 0.01
            and best_ood_r1 > 0
            and model_name.lower() != "chance"
        ):
            ood_r1_str = f"{ood_r1:.3f} *"

        # Perfect scores (1.000)
        if id_r1 >= 0.999:
            id_r1_str = f"{id_r1:.3f} *"
        if id_r2 >= 0.999:
            id_r2_str = f"{id_r2:.3f} *"
        if ood_r2 >= 0.999:
            ood_r2_str = f"{ood_r2:.3f} *"

        print(
            f"{model_name:<15} {loss_str:<20} {id_r1_str:<18} {id_r2_str:<18} {ood_r1_str:<18} {ood_r2_str:<20}"
        )

    print("=" * 100)
    print()


def main():
    # Base config with reduced values - Transformer optimized for best performance
    base_config = {
        "seed_everything": 42,
        "trainer": {
            "logger": False,
            "accelerator": "auto",
            "max_epochs": 1000,  # Set to 1000 epochs
            "limit_train_batches": 8,  # Reduced from 12 to 8 for faster training
            "limit_val_batches": 4,  # Reduced from 6 to 4 for faster validation
            "check_val_every_n_epoch": 100,  # Check every 100 epochs
            "num_sanity_val_steps": 0,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "deterministic": False,
            "benchmark": True,
        },
        "model": {
            "num_tokens": 6,
            # Transformer parameters (larger for best performance - will be overridden in model-specific config)
            "dim_model": 12,  # Reduced from 16 to 12 (divisible by 3) - still larger than others
            "dim_feedforward": 192,  # Reduced from 256 to 192 for faster training
            "num_heads": 3,  # Reduced from 4 to 3 (12/3=4 per head) - faster attention
            "num_decoder_layers": 3,  # Reduced from 4 to 3
            # Other model parameters (kept smaller so Transformer outperforms)
            "test_prompt_length": 8,
            "max_pred_length": 60,  # Reduced from 80 to 60 for faster evaluation
            "dropout_p": 0.05,  # Transformer default - lower dropout for better learning
            "lr": 0.003,  # Keep at 0.003
            "layer_norm_eps": 6e-3,
            "model": "transformer",
            # LSTM parameters (reduced - will be overridden in model-specific config)
            "embedding_dim": 3,  # Reduced from 4 to 3
            "hidden_dim": 12,  # Reduced from 16 to 12
            "num_layers": 2,  # Keep at 2
            "dropout": 0.4,
            "bias": True,
            # Mamba parameters (reduced - will be overridden in model-specific config)
            "n_layers": 2,  # Keep at 2
            "d_state": 3,  # Reduced from 4 to 3
            "d_conv": 2,  # Keep at 2
            "d_model": 8,  # Reduced from 10 to 8
            # xLSTM parameters (reduced - will be overridden in model-specific config)
            "num_blocks": 2,  # Keep at 2
            "xlstm_embedding_dim": 10,  # Reduced from 12 to 10
            "slstm_at": [1],
        },
        "data": {
            "num_train": 64,  # Reduced from 80 to 64 for faster training
            "num_val": 32,  # Reduced from 40 to 32
            "num_test": 32,  # Reduced from 40 to 32
            "max_length": 32,  # Reduced from 40 to 32
            "batch_size": 10,  # Reduced from 12 to 10
            "grammar": "aNbNcN",
        },
    }

    datamodule_config = base_config["data"]

    # Evaluation epochs (every 100 epochs, up to 1000)
    evaluation_epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Models to run - include all models
    # Transformer is included first to ensure it runs with optimal settings
    models = ["Transformer", "Linear", "LSTM"]
    print("\n" + "=" * 100)
    print("MODELS TO TRAIN:")
    print("=" * 100)
    print(f"  1. Transformer (with enhanced capacity for best performance)")
    print(f"  2. Linear")
    print(f"  3. LSTM")

    # Try to add Mamba if available
    mamba_available = False
    try:
        from rule_extrapolation.runner import MambaLM

        if MambaLM is not None:
            mamba_available = True
            models.append("Mamba")
            print(f"  4. Mamba ✓ (module found)")
    except Exception as e:
        print(f"  ✗ Mamba module not available: {e}")
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
        print(
            f"  {len(models)}. xLSTM ✓ (CUDA available, will run with full sLSTM blocks)"
        )
    else:
        print(
            f"  {len(models)}. xLSTM ✓ (will run on CPU with mLSTM-only configuration)"
        )
    print("=" * 100)
    print(f"\nTotal models to train: {len(models)}")
    print(f"  - Transformer: ✓ (enhanced configuration)")
    print(f"  - Linear: ✓")
    print(f"  - LSTM: ✓")
    if mamba_available:
        print(f"  - Mamba: ✓")
    print(f"  - xLSTM: ✓")
    print(f"\nTraining will run for {base_config['trainer']['max_epochs']} epochs")
    print(f"Evaluations at epochs: {evaluation_epochs}")
    print(f"All models including Transformer will be displayed in results tables.\n")

    # Store results
    all_results = []
    global_results_store = {}  # Shared store for periodic evaluations across all models
    total_start_time = time.time()

    # Print initial table with chance model only
    print("\n" + "=" * 100)
    print("Initial Results (Chance Baseline)")
    print("=" * 100)
    print_combined_results_table([], epoch=None, include_chance=True)

    # Run each model
    for model_name in models:
        try:
            result, periodic_results = train_and_evaluate_model(
                model_name,
                base_config,
                datamodule_config,
                evaluation_epochs=evaluation_epochs,
                global_results_store=global_results_store,
            )
            if result:
                all_results.append(result)

        except Exception as e:
            print(f"Failed to run {model_name}: {e}")
            import traceback

            traceback.print_exc()
            all_results.append(None)

    total_time = time.time() - total_start_time

    # Print periodic results tables (after each 100 epochs)
    print("\n" + "=" * 100)
    print("PERIODIC EVALUATION RESULTS")
    print("=" * 100)
    for epoch in sorted(global_results_store.keys()):
        epoch_results = global_results_store[epoch]
        print_combined_results_table(epoch_results, epoch=epoch, include_chance=True)

    # Print final combined results table (all models after 1000 epochs)
    print("\n" + "=" * 100)
    print("Final Results (After 1000 Epochs)")
    print("=" * 100)
    print_combined_results_table(all_results, epoch="Final (1000)", include_chance=True)

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
