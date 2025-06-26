# Adapted from https://github.com/arnab39/equiadapt/blob/main/examples/images/classification/train.py
# MIT License, (c) 2023  (c) 2023 Arnab Mondal

import torch
import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig

from train_utils import (
    get_model_data_and_callbacks,
    get_trainer,
    load_envs,
    get_cifar_dataset,
    setup_hyperparams,
)

from cp.cp_pipelines import run_calibration_trials

def cp(hyperparams: DictConfig) -> None:
    
    _, wandb_logger = setup_hyperparams(hyperparams)

    if not hyperparams.experiment.run_mode == "test":
        raise Exception("run mode must be in test")

    # set seed
    pl.seed_everything(hyperparams.experiment.seed)

    # get model, callbacks, and image data
    model, _, callbacks = get_model_data_and_callbacks(hyperparams)
    
    num_labels = 100 if "100" in hyperparams.dataset.dataset_name else 10

    print(f"Running {hyperparams.dataset.dataset_name} with {hyperparams.cp.score_fn}")

    image_data = get_cifar_dataset(
        hyperparams.dataset, 
        version=num_labels, 
        num_rotations=hyperparams.experiment.inference.num_rotations
    )
   
    # get trainer
    trainer = get_trainer(hyperparams, callbacks, wandb_logger)

    _ = trainer.test(model, datamodule=image_data)

    device = torch.device("cuda") if hyperparams.experiment.device == "cuda" else torch.device("cpu")
    model.to(device)

    results = run_calibration_trials(
        model=model, 
        # method="baseline",
        data_module=image_data, 
        num_resamples=hyperparams.cp.num_resamples,
        score_fn=hyperparams.cp.score_fn,
        num_classes=num_labels,
        alpha=hyperparams.cp.alpha
    )


# load the variables from .env file
load_envs()


@hydra.main(config_path=str("./configs/"), config_name="default")
def main(cfg: DictConfig) -> None:
    cp(cfg)


if __name__ == "__main__":
    main()
