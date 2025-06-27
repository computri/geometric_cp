# Adapted from https://github.com/arnab39/equiadapt/blob/main/examples/pointcloud/classification/train_utils.py
# MIT License, (c) 2023  (c) 2023 Arnab Mondal

import os
from typing import Any, Optional

import dotenv
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import PointcloudClassificationPipeline


def log_model_info(model: Any) -> None:
    """Logs model structure, training configuration, and parameter count."""

    model.train()

    print(model)
    
    rotation_type = getattr(model.hparams.hyperparams.experiment.training, "rotation_type", "unknown")
    print(f"Trained with rotation: {rotation_type}")


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    model.eval()

def setup_hyperparams(hyperparams: DictConfig) -> None:
    hyperparams["canonicalization_type"] = hyperparams["canonicalization"][
        "canonicalization_type"
    ]
    hyperparams["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    hyperparams["dataset"]["data_path"] = (
        hyperparams["dataset"]["data_path"]
        + "/"
        + hyperparams["dataset"]["dataset_name"]
    )
    hyperparams["checkpoint"]["checkpoint_path"] = (
        hyperparams["checkpoint"]["checkpoint_path"]
        + "/"
        + hyperparams["dataset"]["dataset_name"]
        + "/"
        + hyperparams["canonicalization_type"]
        + "/"
        + hyperparams["prediction"]["prediction_network_architecture"]
    )

    # set system environment variables for wandb
    if hyperparams["wandb"]["use_wandb"]:
        print("Using wandb for logging...")
        os.environ["WANDB_MODE"] = "online"
    else:
        print("Wandb disabled for logging...")
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DIR"] = hyperparams["wandb"]["wandb_dir"]
    os.environ["WANDB_CACHE_DIR"] = hyperparams["wandb"]["wandb_cache_dir"]

    # initialize wandb
    wandb.init(
        config=OmegaConf.to_container(hyperparams, resolve=True),  # type: ignore
        entity=hyperparams["wandb"]["wandb_entity"],
        project=hyperparams["wandb"]["wandb_project"],
        dir=hyperparams["wandb"]["wandb_dir"],
    )
    wandb_logger = WandbLogger(
        project=hyperparams["wandb"]["wandb_project"], log_model="all"
    )

    return wandb_logger

def get_model_pipeline(hyperparams: DictConfig) -> pl.LightningModule:

    if hyperparams.experiment.run_mode == "test":
        model = PointcloudClassificationPipeline.load_from_checkpoint(
            checkpoint_path=hyperparams.checkpoint.checkpoint_path
            + "/"
            + hyperparams.checkpoint.checkpoint_name
            + ".ckpt",
            hyperparams=hyperparams,
        )
        canon_params = sum(p.numel() for p in model.canonicalizer.parameters() if p.requires_grad)
        pred_params = sum(p.numel() for p in model.prediction_network.parameters() if p.requires_grad)

        model.freeze()
        model.eval()
    else:
        model = PointcloudClassificationPipeline(hyperparams)

    return model


def get_trainer(
    hyperparams: DictConfig, callbacks: list, wandb_logger: pl.loggers.WandbLogger
) -> pl.Trainer:
    if hyperparams.experiment.run_mode == "dryrun":
        trainer = pl.Trainer(
            fast_dev_run=5,
            max_epochs=hyperparams.experiment.training.num_epochs,
            accelerator="auto",
            limit_train_batches=5,
            limit_val_batches=5,
            logger=wandb_logger,
            callbacks=callbacks,
            enable_model_summary=True,
            deterministic=hyperparams.experiment.deterministic,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=hyperparams.experiment.training.num_epochs,
            accelerator="auto",
            logger=wandb_logger,
            callbacks=callbacks,
            deterministic=hyperparams.experiment.deterministic,
            num_nodes=hyperparams.experiment.num_nodes,
            devices=hyperparams.experiment.num_gpus,
            enable_model_summary=True,
            strategy="ddp",
        )

    return trainer



def get_recursive_hyperparams_identifier(hyperparams: DictConfig) -> str:
    # get the identifier for the canonicalization network hyperparameters
    # recursively go through the dictionary and get the values and concatenate them
    identifier = ""
    for key, value in hyperparams.items():
        if isinstance(value, DictConfig):
            identifier += f"_{get_recursive_hyperparams_identifier(value)}_"  # type: ignore
        else:
            identifier += f"_{key}_{value}_"  # type: ignore
    return identifier


def get_checkpoint_name(hyperparams: DictConfig) -> str:

    return (
        f"{get_recursive_hyperparams_identifier(hyperparams.canonicalization)}".lstrip(
            "_"
        )
        + f"__epochs_{hyperparams.experiment.training.num_epochs}_"
        + f"__seed_{hyperparams.experiment.seed}"
    )


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)
