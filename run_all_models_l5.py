#!/usr/bin/env python3
"""
Script to train and test all models (Transformer, Linear, LSTM, Mamba, xLSTM) 
for aNbNcN grammar (L5) and display results in a combined table format.
"""

import sys
import os
import torch
import math
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.utilities.seed import seed_everything
from omegaconf import OmegaConf
import time
from collections import defaultdict

from rule_extrapolation.runner import LightningGrammarModule
from rule_extrapolation.datamodule import GrammarDataModule
from rule_extrapolation.data import (
    SOS_token,
    A_token,
    B_token,
    C_token,
    check_same_number_as_bs_cs,
    check_as_before_bs_before_cs,
)


def convert_to_float(value):
    """Convert tensor or other types to float."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def calc_distances_anbncn(prompt):
    """Calculate distances for aNbNcN grammar (N1, N2) where N1 and N2 are the two smaller distances."""
    num_3 = (prompt == A_token.item()).sum().item()
    num_4 = (prompt == B_token.item()).sum().item()
    num_5 = (prompt == C_token.item()).sum().item()
    max_num = max(num_3, num_4, num_5)
    distances = sorted([max_num - num_3, max_num - num_4, max_num - num_5])
    distances = distances[1:]  # Remove the smallest (which is 0)
    return tuple(distances)


def r1_formula_anbncn(N1, N2, n):
    """Calculate R1 chance level accuracy for aNbNcN."""
    tot = 0.0
    for m in range(0, (n - (N1 + N2)) // 3 + 1):
        tot += (
            math.comb(N1 + N2 + 3 * m, N1 + m)
            * math.comb(N2 + 2 * m, m)
            * (1 / (4 ** (N1 + N2 + 3 * m + 1)))
        )
    return tot


def r2_formula_anbncn(prompt, test_prompt_length, max_pred_length):
    """Calculate R2 chance level accuracy for aNbNcN."""
    n = max_pred_length - len(prompt[1:])
    acc = 0.0
    if prompt[-1] == C_token.item():
        for i in range(0, n + 1):
            acc += 1 / (4 ** (i + 1))
    elif prompt[-1] == B_token.item():
        for i in range(0, n + 1):
            for b in range(0, i + 1):
                acc += 1 / (4 ** (i + 1))
    elif prompt[-1] == A_token.item():
        for i in range(0, n + 1):
            for a in range(0, i + 1):
                for b in range(0, i - a + 1):
                    acc += 1 / (4 ** (i + 1))
    else:
        # SOS token or other
        for i in range(0, n + 1):
            for a in range(0, i + 1):
                for b in range(0, i - a + 1):
                    acc += 1 / (4 ** (i + 1))
    return acc


def r2_ood_anbncn(len_n_list):
    """Calculate OOD R2 chance level accuracy for aNbNcN."""
    tot = 0.0
    all_count = 0
    for n, number in len_n_list:
        all_count += number
        for i in range(0, n + 1):
            for a in range(0, i + 1):
                for b in range(0, i - a + 1):
                    tot += (1 / (4 ** (i + 1))) * number
    return tot / all_count if all_count > 0 else 0.0


def calculate_chance_level_metrics(
    test_prompts_id, test_prompts_ood, test_prompt_length, max_pred_length
):
    """Calculate chance level metrics for aNbNcN grammar."""
    # Calculate ID R1
    my_dict_id = defaultdict(int)
    for prompt in test_prompts_id:
        n = max_pred_length - len(prompt[1:])
        distances = calc_distances_anbncn(prompt)
        distances += (n,)
        my_dict_id[distances] += 1

    id_r1_acc = 0.0
    for distances in my_dict_id.keys():
        N1, N2, n = distances
        id_r1_acc += r1_formula_anbncn(N1, N2, n) * my_dict_id[distances]
    id_r1_acc /= sum(my_dict_id.values()) if my_dict_id else 1.0

    # Calculate OOD R1
    my_dict_ood = defaultdict(int)
    len_n_list = []
    for prompt in test_prompts_ood:
        n = max_pred_length - len(prompt[1:])
        distances = calc_distances_anbncn(prompt)
        distances += (n,)
        my_dict_ood[distances] += 1
        # Track unique (n, count) pairs
        found = False
        for idx, (existing_n, count) in enumerate(len_n_list):
            if existing_n == n:
                len_n_list[idx] = (n, count + 1)
                found = True
                break
        if not found:
            len_n_list.append((n, 1))

    ood_r1_acc = 0.0
    for distances in my_dict_ood.keys():
        N1, N2, n = distances
        ood_r1_acc += r1_formula_anbncn(N1, N2, n) * my_dict_ood[distances]
    ood_r1_acc /= sum(my_dict_ood.values()) if my_dict_ood else 1.0

    # Calculate ID R2
    id_r2_accuracy = []
    for prompt in test_prompts_id:
        r2 = r2_formula_anbncn(prompt, test_prompt_length, max_pred_length)
        id_r2_accuracy.append(r2)
    id_r2_acc = sum(id_r2_accuracy) / len(id_r2_accuracy) if id_r2_accuracy else 0.0

    # Calculate OOD R2
    ood_r2_acc = r2_ood_anbncn(len_n_list)

    return id_r1_acc, id_r2_acc, ood_r1_acc, ood_r2_acc


class EpochEvaluationCallback(Callback):
    """Callback to evaluate and store results every N epochs."""

    def __init__(
        self,
        model_name,
        datamodule,
        epoch_interval=50,
        results_dict=None,
        chance_metrics=None,
        all_models=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.datamodule = datamodule
        self.epoch_interval = epoch_interval
        self.results_dict = results_dict if results_dict is not None else {}
        self.chance_metrics = chance_metrics
        self.all_models = all_models if all_models is not None else []

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch, evaluate at checkpoint intervals."""
        current_epoch = trainer.current_epoch
        # Evaluate at epochs 0, 50, 100, 150, etc. (every 50 epochs)
        # Note: epoch numbers are 0-indexed, so epoch 0 is the first epoch
        if current_epoch % self.epoch_interval == 0:
            self._evaluate_and_store(trainer, pl_module, current_epoch)

    def _evaluate_and_store(self, trainer, pl_module, epoch):
        """Evaluate the model and store results."""
        model = pl_module
        model.eval()
        model.freeze()

        # Setup test datamodule if not already done
        if not hasattr(self, "_test_setup_done"):
            self.datamodule.setup("test")
            self._test_setup_done = True

        test_dl = self.datamodule.test_dataloader()

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
                    print(
                        f"Error calculating loss for {self.model_name} at epoch {epoch}: {e}"
                    )
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
            print(
                f"Error evaluating prompts for {self.model_name} at epoch {epoch}: {e}"
            )
            id_r1 = id_r2 = ood_r1 = ood_r2_completion = 0.0

        # Store results
        if epoch not in self.results_dict:
            self.results_dict[epoch] = []

        result = {
            "model": self.model_name,
            "test_loss": test_loss,
            "id_r1": id_r1,
            "id_r2": id_r2,
            "ood_r1": ood_r1,
            "ood_r2_completion": ood_r2_completion,
            "epoch": epoch,
        }

        self.results_dict[epoch].append(result)

        # Print combined results table for this epoch
        # Get all results for this epoch
        epoch_results = self.results_dict.get(epoch, [])

        # Print the table using the existing function
        # This will show all models that have completed evaluation at this epoch
        print(f"\n{'='*100}")
        print(f"EPOCH {epoch} CHECKPOINT - Results for all models evaluated so far")
        print(f"{'='*100}")
        print_combined_results_table(epoch_results, self.chance_metrics, epoch=epoch)

        # Unfreeze for continued training
        model.unfreeze()
        model.train()


def train_and_evaluate_model_single_run(
    model_name,
    config_base,
    datamodule_config,
    seed,
    results_dict=None,
    chance_metrics=None,
    run_idx=0,
    num_runs=1,
):
    """Train and evaluate a single model for one run with a specific seed."""
    if num_runs > 1:
        print(f"\n{'='*80}")
        print(
            f"Training {model_name.upper()} model - Run {run_idx+1}/{num_runs} (seed={seed})..."
        )
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()} model...")
        print(f"{'='*80}")

    start_time = time.time()

    if results_dict is None:
        results_dict = {}

    # Set seed for this run
    seed_everything(seed, workers=True)
    config_base["seed_everything"] = seed

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

    # Create epoch evaluation callback (only for single runs to save time)
    callbacks_list = [checkpoint_callback]
    if num_runs == 1 and results_dict is not None:
        epoch_eval_callback = EpochEvaluationCallback(
            model_name=model_name,
            datamodule=datamodule,
            epoch_interval=50,
            results_dict=results_dict,
            chance_metrics=chance_metrics,
            all_models=None,
        )
        callbacks_list.append(epoch_eval_callback)

    # Get trainer config with defaults
    accelerator = config.trainer.get("accelerator", "gpu")
    devices = config.trainer.get("devices", 1)
    precision = config.trainer.get("precision", "16-mixed")

    # Ensure devices is valid for the accelerator
    if accelerator == "cpu" and devices is None:
        devices = 1  # CPU requires devices to be an int > 0

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=config.trainer.max_epochs,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        callbacks=callbacks_list,
        logger=config.trainer.get("logger", False),
        enable_progress_bar=config.trainer.get("enable_progress_bar", True),
        enable_model_summary=config.trainer.get("enable_model_summary", False),
        num_sanity_val_steps=config.trainer.get("num_sanity_val_steps", 0),
        deterministic=config.trainer.get("deterministic", False),
        benchmark=config.trainer.get("benchmark", True),
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
    if num_runs > 1:
        print(
            f"{model_name} run {run_idx+1}/{num_runs} completed in {elapsed_time:.1f} seconds"
        )
    else:
        print(f"{model_name} completed in {elapsed_time:.1f} seconds")

    return {
        "model": model_name,
        "test_loss": test_loss,
        "id_r1": id_r1,
        "id_r2": id_r2,
        "ood_r1": ood_r1,
        "ood_r2_completion": ood_r2_completion,
        "time": elapsed_time,
        "seed": seed,
    }


def train_and_evaluate_model(
    model_name,
    config_base,
    datamodule_config,
    results_dict=None,
    chance_metrics=None,
    num_runs=1,
    seeds=None,
):
    """Train and evaluate a model (single or multiple runs with different seeds)."""
    if seeds is None:
        # Default seeds for reproducibility
        seeds = [42 + i for i in range(num_runs)]

    if num_runs == 1:
        # Single run - use the old behavior with epoch callbacks
        return train_and_evaluate_model_single_run(
            model_name,
            config_base,
            datamodule_config,
            seeds[0],
            results_dict,
            chance_metrics,
            0,
            1,
        )
    else:
        # Multiple runs - collect results and calculate statistics
        all_runs_results = []
        total_start_time = time.time()

        for run_idx, seed in enumerate(seeds):
            result = train_and_evaluate_model_single_run(
                model_name,
                config_base,
                datamodule_config,
                seed,
                None,
                chance_metrics,
                run_idx,
                num_runs,
            )
            if result:
                all_runs_results.append(result)

        total_time = time.time() - total_start_time

        if len(all_runs_results) == 0:
            return None

        # Calculate statistics
        stats = calculate_statistics(all_runs_results)

        # Return aggregated result with statistics
        return {
            "model": model_name,
            "test_loss": stats["test_loss"]["mean"] if stats else float("inf"),
            "id_r1": stats["id_r1"]["mean"] if stats else 0.0,
            "id_r2": stats["id_r2"]["mean"] if stats else 0.0,
            "ood_r1": stats["ood_r1"]["mean"] if stats else 0.0,
            "ood_r2_completion": stats["ood_r2_completion"]["mean"] if stats else 0.0,
            "time": total_time,
            "stats": stats,
            "all_runs": all_runs_results,
        }


def calculate_statistics(results_list):
    """Calculate mean and std for multiple runs of the same model."""
    if not results_list or len(results_list) == 0:
        return None

    metrics = ["test_loss", "id_r1", "id_r2", "ood_r1", "ood_r2_completion"]
    stats = {}

    for metric in metrics:
        values = [r[metric] for r in results_list if r is not None and metric in r]
        values = [v for v in values if v != float("inf") and v != float("-inf")]

        if len(values) > 0:
            stats[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values,
            }
        else:
            stats[metric] = {"mean": float("inf"), "std": 0.0, "values": []}

    return stats


def find_best_values(all_results_stats):
    """Find the best (highest) value for each metric across all models."""
    metrics = ["test_loss", "id_r1", "id_r2", "ood_r1", "ood_r2_completion"]
    best = {}

    for metric in metrics:
        best_value = None
        best_model = None

        for model_name, stats in all_results_stats.items():
            if stats is None or metric not in stats:
                continue

            mean_val = stats[metric]["mean"]
            if mean_val == float("inf") or mean_val == float("-inf"):
                continue

            # For test_loss, lower is better; for others, higher is better
            if metric == "test_loss":
                if best_value is None or mean_val < best_value:
                    best_value = mean_val
                    best_model = model_name
            else:
                if best_value is None or mean_val > best_value:
                    best_value = mean_val
                    best_model = model_name

        best[metric] = {"model": best_model, "value": best_value}

    return best


def format_value_with_std(mean, std, is_best=False, metric="test_loss"):
    """Format a value with standard deviation, optionally marking as best."""
    if mean == float("inf") or mean == float("-inf"):
        return "N/A"

    # Round to 3 decimal places
    mean = round(mean, 3)
    std = round(std, 3)

    # Format based on metric - if std is very small (< 0.001), show mean only
    if std < 0.001:
        formatted = f"{mean:.3f}"
    else:
        formatted = f"{mean:.3f} ± {std:.3f}"

    # Mark best values - in terminal we'll use asterisks for visibility
    # (In the image they're bolded, but terminal doesn't support bold easily)
    if is_best and metric != "test_loss":
        # For non-loss metrics, higher is better - mark with asterisks
        return f"**{formatted}**"

    return formatted


def print_combined_results_table(
    results_list,
    chance_metrics=None,
    epoch=None,
    all_results_stats=None,
    best_values=None,
):
    """Print combined results table for all models."""
    epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
    print("\n" + "=" * 100)
    print(
        f"Table 6: Test loss and rule-following accuracies for the context-sensitive language L5 = {{a^n b^n c^n}}: the Transformer can extrapolate (R1) the best{epoch_str}"
    )
    print("=" * 100)
    print(
        f"{'Model':<15} {'Test loss':<20} {'ID R1':<15} {'ID R2':<15} {'OOD R1':<15} {'OOD R2 completion':<20}"
    )
    print("-" * 100)

    # Add Chance as first row if available
    if chance_metrics:
        id_r1 = round(chance_metrics["id_r1"], 3)
        id_r2 = round(chance_metrics["id_r2"], 3)
        ood_r1 = round(chance_metrics["ood_r1"], 3)
        ood_r2 = round(chance_metrics["ood_r2_completion"], 3)
        print(
            f"{'Chance':<15} {'N/A':<20} {id_r1:.3f}            {id_r2:.3f}            {ood_r1:.3f}            {ood_r2:.3f}"
        )

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

        # Check if we have statistics (multiple runs) or single result
        if (
            all_results_stats
            and model_name in all_results_stats
            and all_results_stats[model_name] is not None
        ):
            # Use statistics from multiple runs
            stats = all_results_stats[model_name]

            # Check if this model has the best value for each metric
            is_best_test_loss = (
                best_values
                and best_values.get("test_loss", {}).get("model") == model_name
            )
            is_best_id_r1 = (
                best_values and best_values.get("id_r1", {}).get("model") == model_name
            )
            is_best_id_r2 = (
                best_values and best_values.get("id_r2", {}).get("model") == model_name
            )
            is_best_ood_r1 = (
                best_values and best_values.get("ood_r1", {}).get("model") == model_name
            )
            is_best_ood_r2 = (
                best_values
                and best_values.get("ood_r2_completion", {}).get("model") == model_name
            )

            loss_str = format_value_with_std(
                stats["test_loss"]["mean"],
                stats["test_loss"]["std"],
                is_best_test_loss,
                "test_loss",
            )
            id_r1_str = format_value_with_std(
                stats["id_r1"]["mean"], stats["id_r1"]["std"], is_best_id_r1, "id_r1"
            )
            id_r2_str = format_value_with_std(
                stats["id_r2"]["mean"], stats["id_r2"]["std"], is_best_id_r2, "id_r2"
            )
            ood_r1_str = format_value_with_std(
                stats["ood_r1"]["mean"],
                stats["ood_r1"]["std"],
                is_best_ood_r1,
                "ood_r1",
            )
            ood_r2_str = format_value_with_std(
                stats["ood_r2_completion"]["mean"],
                stats["ood_r2_completion"]["std"],
                is_best_ood_r2,
                "ood_r2_completion",
            )
        else:
            # Single result (no statistics)
            test_loss = round(model_results["test_loss"], 3)
            id_r1 = round(model_results["id_r1"], 3)
            id_r2 = round(model_results["id_r2"], 3)
            ood_r1 = round(model_results["ood_r1"], 3)
            ood_r2 = round(model_results["ood_r2_completion"], 3)

            if test_loss == float("inf") or test_loss == float("-inf"):
                loss_str = "N/A"
            else:
                loss_str = f"{test_loss:.3f}"

            id_r1_str = f"{id_r1:.3f}"
            id_r2_str = f"{id_r2:.3f}"
            ood_r1_str = f"{ood_r1:.3f}"
            ood_r2_str = f"{ood_r2:.3f}"

        print(
            f"{model_name:<15} {loss_str:<20} {id_r1_str:<15} {id_r2_str:<15} {ood_r1_str:<15} {ood_r2_str:<20}"
        )

    print("=" * 100)
    print()


def main():
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(
            f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n"
        )
        accelerator = "gpu"
        devices = 1
        precision = "16-mixed"
    else:
        print("⚠ Warning: No GPU detected. Training will be slower on CPU.")
        print("  For faster training, use Google Colab with GPU runtime.")
        print("  The script will continue with CPU...\n")
        accelerator = "cpu"
        devices = 1  # CPU requires devices to be an int > 0, not None
        precision = "32"

    # Base config with increased values for better accuracy
    base_config = {
        "seed_everything": 42,
        "trainer": {
            "logger": False,
            "accelerator": accelerator,
            "devices": devices,
            "max_epochs": 1000,
            "limit_train_batches": None,
            "limit_val_batches": None,
            "check_val_every_n_epoch": 50,
            "num_sanity_val_steps": 0,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "deterministic": False,
            "benchmark": True,
            "precision": precision,
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

    # Add xLSTM (requires GPU for full sLSTM blocks)
    models.append("xLSTM")
    if torch.cuda.is_available():
        print("✓ GPU available, will run xLSTM with full sLSTM blocks")
    else:
        print("⚠ Warning: No GPU detected. xLSTM may not work properly without CUDA.")
        print("  Consider using Google Colab with GPU runtime for optimal performance.")

    # Calculate chance level metrics
    print("Calculating chance level metrics...")
    chance_metrics = None
    try:
        # Create a dummy model to get test prompts (we only need the prompts, not training)
        dummy_config = OmegaConf.create(base_config.copy())
        dummy_config.model.model = "transformer"
        dummy_datamodule = GrammarDataModule(
            num_train=datamodule_config["num_train"],
            num_val=datamodule_config["num_val"],
            num_test=datamodule_config["num_test"],
            max_length=datamodule_config["max_length"],
            batch_size=datamodule_config["batch_size"],
            grammar=datamodule_config["grammar"],
        )
        dummy_datamodule.prepare_data()
        dummy_datamodule.setup("fit")

        # Create model just to get test prompts
        # Use CPU device to avoid GPU requirements
        device = torch.device("cpu")
        dummy_model = LightningGrammarModule(
            num_tokens=6,
            test_prompt_length=base_config["model"]["test_prompt_length"],
            max_pred_length=base_config["model"]["max_pred_length"],
            lr=base_config["model"]["lr"],
            model="transformer",
            grammar=datamodule_config["grammar"],
            max_data_length=datamodule_config["max_length"],
            batch_size=datamodule_config["batch_size"],
            dim_model=8,
            dim_feedforward=128,
            num_heads=4,
            num_decoder_layers=2,
            dropout_p=0.1,
            layer_norm_eps=6e-3,
        )
        dummy_model.to(device)
        dummy_model.hparams.device = device

        # Get test prompts (they should be set up during model initialization)
        test_prompts_id = dummy_model.test_prompts_in_distribution.cpu()
        test_prompts_ood = dummy_model.test_prompts_out_of_distribution.cpu()

        if len(test_prompts_id) == 0 or len(test_prompts_ood) == 0:
            raise ValueError(
                "No test prompts found. Model may not have initialized correctly."
            )

        # Calculate chance metrics
        id_r1, id_r2, ood_r1, ood_r2 = calculate_chance_level_metrics(
            test_prompts_id,
            test_prompts_ood,
            base_config["model"]["test_prompt_length"],
            base_config["model"]["max_pred_length"],
        )

        chance_metrics = {
            "model": "Chance",
            "test_loss": float("inf"),  # N/A for chance
            "id_r1": id_r1,
            "id_r2": id_r2,
            "ood_r1": ood_r1,
            "ood_r2_completion": ood_r2,
        }
        print(f"Chance level metrics calculated:")
        print(f"  ID R1: {id_r1:.6f}")
        print(f"  ID R2: {id_r2:.6f}")
        print(f"  OOD R1: {ood_r1:.6f}")
        print(f"  OOD R2: {ood_r2:.6f}\n")
    except Exception as e:
        print(f"Warning: Could not calculate chance metrics: {e}")
        print("Continuing without chance metrics...\n")

    # Configuration for multiple runs (for statistics)
    NUM_RUNS = 3  # Number of runs per model for statistics (use 1 for single run, 3-5 for stats)
    USE_EPOCH_CALLBACKS = (
        NUM_RUNS == 1
    )  # Only show epoch-by-epoch results for single runs

    # Reduce epochs for faster execution while maintaining good results
    # Original was 1000, but 500 should be sufficient for convergence
    if NUM_RUNS > 1:
        # For multiple runs, use fewer epochs per run to save time
        base_config["trainer"]["max_epochs"] = 500
        print(
            f"Using {base_config['trainer']['max_epochs']} epochs per run for faster execution with {NUM_RUNS} runs"
        )
    else:
        # For single run, keep original epochs
        base_config["trainer"]["max_epochs"] = 1000

    # Store results - shared dictionary to track results across epochs (only for single runs)
    results_dict = (
        {} if USE_EPOCH_CALLBACKS else None
    )  # {epoch: [result1, result2, ...]}
    all_results = []
    all_results_with_stats = {}  # Store results with statistics for each model
    total_start_time = time.time()

    # Run each model sequentially
    print(f"\n{'='*100}")
    print(f"Running {NUM_RUNS} run(s) per model for statistics")
    print(f"{'='*100}\n")

    for model_name in models:
        try:
            result = train_and_evaluate_model(
                model_name,
                base_config,
                datamodule_config,
                results_dict=results_dict,
                chance_metrics=chance_metrics,
                num_runs=NUM_RUNS,
                seeds=None,  # Will use default seeds
            )
            if result:
                all_results.append(result)
                # Store statistics if available
                if "stats" in result and result["stats"]:
                    all_results_with_stats[model_name] = result["stats"]
        except Exception as e:
            print(f"Failed to run {model_name}: {e}")
            all_results.append(None)

    total_time = time.time() - total_start_time

    # Calculate best values across all models
    best_values = (
        find_best_values(all_results_with_stats) if all_results_with_stats else None
    )

    # Print epoch checkpoints summary (only for single runs)
    if USE_EPOCH_CALLBACKS and results_dict:
        print("\n" + "=" * 100)
        print("SUMMARY OF ALL EPOCH CHECKPOINTS")
        print("=" * 100)
        sorted_epochs = sorted(results_dict.keys())
        if sorted_epochs:
            print(
                f"\nResults were printed at epochs: {', '.join(map(str, sorted_epochs))}"
            )
            print("See above for detailed tables at each checkpoint.\n")

    # Print final combined results table with statistics
    print("\n" + "=" * 100)
    print("FINAL RESULTS (After All Training)")
    if NUM_RUNS > 1:
        print(f"Results shown as mean ± std from {NUM_RUNS} runs per model")
        print("**Bold values** indicate best performance for that metric")
    print("=" * 100)
    print_combined_results_table(
        all_results,
        chance_metrics,
        epoch=None,
        all_results_stats=all_results_with_stats,
        best_values=best_values,
    )

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
