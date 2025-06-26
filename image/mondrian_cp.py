# Adapted from https://github.com/arnab39/equiadapt/blob/main/examples/images/classification/train.py
# MIT License, (c) 2023  (c) 2023 Arnab Mondal



import torch
import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig

from train_utils import (
    get_model_pipeline,
    load_envs,
    setup_hyperparams,
    get_cifar_dataset
)
from cp.cp_pipelines import run_mcp_trials
import info_utils


def mcp(hyperparams: DictConfig) -> None:
    
    _ = setup_hyperparams(hyperparams)
    
    # simpler naming
    target_partition = hyperparams.cp.mondrian.partitioning
    shift_type = hyperparams.dataset.cp_experiments.shift_type

    log_dir = f"./experiment_logs/mondrian/{target_partition}/{shift_type}/"

    # save info meta-data to file, used for partitioning
    getattr(info_utils, f"get_{target_partition}_info")(path=log_dir)    

    if not hyperparams.experiment.run_mode == "test":
        raise Exception("run mode must be in test")

    # set seed
    pl.seed_everything(hyperparams.experiment.seed)

    # get model
    model = get_model_pipeline(hyperparams)

    print(f"Running {hyperparams.dataset.dataset_name} with {hyperparams.cp.score_fn}")
   
    image_data = get_cifar_dataset(
        hyperparams.dataset,
        version=model.num_classes,
        num_rotations=hyperparams.experiment.inference.num_rotations,
        partition=hyperparams.cp.mondrian.partitioning
    )
    image_data.setup(stage="test")


    device = torch.device("cuda") if hyperparams.experiment.device == "cuda" else torch.device("cpu")
    model.to(device)

    print("--------- MCP ---------")
    _ = run_mcp_trials(
        model=model, 
        method="mondrian",
        data_module=image_data,
        num_resamples=hyperparams.cp.num_resamples,
        score_fn=hyperparams.cp.score_fn,
        num_classes=model.num_classes,
        alpha=hyperparams.cp.alpha,
        print_class_results=True,
        print_partition_results=True,
        save_results=True,
        canon_cutoff=hyperparams.cp.canon_cutoff,
        partitioning=hyperparams.cp.mondrian.partitioning,
        log_dir=log_dir

    )

    print("--------- SCP ---------")
    _ = run_mcp_trials(
        model=model, 
        method="baseline",
        data_module=image_data,
        num_resamples=hyperparams.cp.num_resamples,
        score_fn=hyperparams.cp.score_fn,
        num_classes=model.num_classes,
        alpha=hyperparams.cp.alpha,
        save_results=True,
        print_class_results=True,
        partitioning=hyperparams.cp.mondrian.partitioning,
        log_dir=log_dir
    )


# load the variables from .env file
load_envs()


@hydra.main(config_path=str("./configs/"), config_name="default")
def main(cfg: DictConfig) -> None:
    mcp(cfg)


if __name__ == "__main__":
    main()
