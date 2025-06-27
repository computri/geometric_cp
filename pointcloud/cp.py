# Adapted from https://github.com/arnab39/equiadapt/blob/main/examples/pointcloud/classification/train.py
# MIT License, (c) 2023  (c) 2023 Arnab Mondal


from typing import Any

import hydra
import torch
import wandb
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from cp.cp_pipelines import run_calibration_trials
from prepare import ModelNetDataModule
from train_utils import (
    get_model_pipeline,
    get_trainer,
    load_envs,
    log_model_info,
    setup_hyperparams,
)

def test_and_cp(
    hyperparams: DictConfig, 
    model: Any, 
    wandb_logger: WandbLogger,
    datamodule: Any,
    device) -> None:

    # get trainer
    trainer = get_trainer(hyperparams, None, wandb_logger)


    # Report accuracy
    _ = trainer.test(model, datamodule=datamodule)


    datamodule.setup(stage="test")

    device = torch.device("cuda") if hyperparams.experiment.device == "cuda" else torch.device("cpu")
    model.to(device)

    results = run_calibration_trials(
        model=model, 
        data_module=datamodule,
        num_resamples=hyperparams.cp.num_resamples,
        score_fn=hyperparams.cp.score_fn,
        num_classes=40,
        alpha=hyperparams.cp.alpha
    )


def cp(hyperparams: DictConfig) -> None:

    wandb_logger = setup_hyperparams(hyperparams)

    # set seed
    pl.seed_everything(hyperparams.experiment.seed)

    # get pointcloud data
    pointcloud_data = ModelNetDataModule(hyperparams.dataset)

    # get model pipeline
    model = get_model_pipeline(hyperparams)

    if hyperparams.canonicalization_type in ("group_equivariant", "opt_equivariant"):
        wandb.watch(model.canonicalizer.canonicalization_network, log="all")
    
    device = torch.device("cuda") if hyperparams.experiment.device == "cuda" else torch.device("cpu")
    
    test_and_cp(
        hyperparams=hyperparams,
        model=model,
        wandb_logger=wandb_logger,
        datamodule=pointcloud_data,
        device=device
    )


# load the variables from .env file
load_envs()


@hydra.main(config_path=str("./configs/"), config_name="default")
def main(cfg: DictConfig) -> None:
    cp(cfg)


if __name__ == "__main__":
    main()
