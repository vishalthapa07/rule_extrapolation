from rule_extrapolation.cli import LLMLightningCLI
from rule_extrapolation.runner import LightningGrammarModule
from rule_extrapolation.datamodule import GrammarDataModule
from os.path import abspath, dirname, join


def test_cli_fast_dev_run():
    # Build args without config file to avoid logger type adaptation issues
    args = [
        "fit",
        "--trainer.fast_dev_run",
        "true",
        "--trainer.logger",
        "false",
        "--trainer.max_epochs",
        "1",
        "--model.num_tokens",
        "6",
        "--model.dim_model",
        "8",
        "--model.dim_feedforward",
        "128",
        "--model.num_heads",
        "4",
        "--model.num_decoder_layers",
        "2",
        "--model.test_prompt_length",
        "6",
        "--model.max_pred_length",
        "32",
        "--model.dropout_p",
        "0.1",
        "--model.lr",
        "0.01",
        "--model.layer_norm_eps",
        "2e-4",
        "--model.model",
        "transformer",
        "--data.num_train",
        "32",
        "--data.num_val",
        "16",
        "--data.num_test",
        "8",
        "--data.max_length",
        "4",
        "--data.batch_size",
        "8",
        "--data.grammar",
        "aNbN",
    ]
    cli = LLMLightningCLI(
        LightningGrammarModule,
        GrammarDataModule,
        save_config_callback=None,
        run=True,
        args=args,
        parser_kwargs={"parse_as_dict": False},
    )
