# Adapted from https://github.com/arnab39/equiadapt/blob/main/examples/images/classification/train_utils.py
# MIT License, (c) 2023  (c) 2023 Arnab Mondal


from typing import Optional

import dotenv
import pytorch_lightning as pl
from model import ImageClassifierPipeline
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import wandb
import os
from prepare import (
    CIFAR10DataModule,
    CIFAR100DataModule,
    ClassConditionalCIFARDataModule
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch

from cp.cp_utils import get_num_bins


def setup_hyperparams(hyperparams):
    hyperparams["experiment"]["run_mode"] = "test"

    assert (
        len(hyperparams["checkpoint"]["checkpoint_name"]) > 0
    ), "checkpoint_name must be provided for test mode"
    hyperparams["checkpoint"]["checkpoint_path"] += f'/{hyperparams["dataset"]["dataset_name"]}'
    existing_ckpt_path = (
        hyperparams["checkpoint"]["checkpoint_path"] 
        + "/"
        + hyperparams["checkpoint"]["checkpoint_name"]
        + ".ckpt"
    )
    existing_ckpt = torch.load(existing_ckpt_path)

    conf = OmegaConf.create(existing_ckpt["hyper_parameters"]["hyperparams"])
    
    hyperparams["canonicalization_type"] = conf["canonicalization_type"]
    hyperparams["canonicalization"] = conf["canonicalization"]
    if hyperparams["checkpoint"]["strict_loading"]:
        hyperparams["prediction"] = conf["prediction"]

    
    # set system environment variables for wandb
    if hyperparams["wandb"]["use_wandb"]:
        print("Using wandb for logging...")
        os.environ["WANDB_MODE"] = "online"
    else:
        print("Wandb disabled for logging...")
        print(hyperparams["wandb"]["wandb_dir"])

        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DIR"] = hyperparams["wandb"]["wandb_dir"]
    os.environ["WANDB_CACHE_DIR"] = hyperparams["wandb"]["wandb_cache_dir"]
        # initialize wandb
    wandb_run = wandb.init(
        config=OmegaConf.to_container(hyperparams, resolve=True),
        entity=hyperparams["wandb"]["wandb_entity"],
        project=hyperparams["wandb"]["wandb_project"],
        dir=hyperparams["wandb"]["wandb_dir"],
    )
    wandb_logger = WandbLogger(
        project=hyperparams["wandb"]["wandb_project"], 
        log_model=False,
        offline=True
    )
    
    return wandb_run, wandb_logger


def get_model_pipeline(hyperparams: DictConfig) -> pl.LightningModule:

    if hyperparams.experiment.run_mode == "test":
        model = ImageClassifierPipeline.load_from_checkpoint(
            checkpoint_path=hyperparams.checkpoint.checkpoint_path
            + "/"
            + hyperparams.checkpoint.checkpoint_name
            + ".ckpt",
            hyperparams=hyperparams,
            strict=hyperparams.checkpoint.strict_loading,
        )
        
        model.freeze()
        model.eval()
    else:
        model = ImageClassifierPipeline(hyperparams)

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
            strategy="ddp",
        )

    return trainer

def get_augment_mode(ckpt_name: str) -> str | None:
    """Infer augmentation type based on checkpoint naming convention."""
    if "canon_c4" in ckpt_name:
        return "c4"
    elif "canon_c8" in ckpt_name:
        return "c8"
    return None

def get_image_data(dataset_hyperparams: DictConfig) -> pl.LightningDataModule:

    dataset_classes = {
        "cifar10": CIFAR10DataModule,
        "cifar100": CIFAR100DataModule,
    }

    if dataset_hyperparams.dataset_name not in dataset_classes:
        raise ValueError(f"{dataset_hyperparams.dataset_name} not implemented")

    return dataset_classes[dataset_hyperparams.dataset_name](dataset_hyperparams)


def discretized_gaussian(size, mean=None, std_dev=None):
    if mean is None:
        mean = size // 2
    if std_dev is None:
        std_dev = size / 4
    
    x = np.linspace(0, size - 1, size)
    distribution = np.exp(-(x - mean)**2 / (2 * std_dev**2))
    return distribution / distribution.sum()


def get_cifar_dataset(hyperparams, partition="label", version=100, num_rotations=8):
    if hyperparams.cp_experiments.class_conditional_joint:

        
        num_partitions = get_num_bins(partition, version)

        std_dev_dict = {
            "dirac": [0.00001] * num_partitions,
            "normal": [1.0] * num_partitions,
            "var-gauss": [0.0001] * int((num_partitions - 6)/ 2) + [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] + [10.0] * int((num_partitions - 6)/ 2)
        }
        
        std_dev = std_dev_dict[hyperparams.cp_experiments.shift_type]

        
        partition_distributions = {idx: discretized_gaussian(num_rotations, std_dev=std_dev[idx]) for idx in range(num_partitions)}
        
        # gaussian centered at consecutive elements
        for i, val in enumerate(partition_distributions.values()):
            partition_distributions[i] = np.roll(val, i)
            if i == num_rotations-1:
                break

        print("using class conditional data...")
        image_data = ClassConditionalCIFARDataModule(
            hyperparams,
            version=version,
            num_rotations=num_rotations,
            partition=partition,
            partition_distributions=partition_distributions
        )
        return image_data


    else:
        image_data = get_image_data(hyperparams)
        return image_data




def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)
