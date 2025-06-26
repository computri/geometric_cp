# Adapted from https://github.com/arnab39/equiadapt/blob/main/examples/images/classification/train.py
# MIT License, (c) 2023  (c) 2023 Arnab Mondal

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from train_utils import (
    load_envs, 
    get_image_data, 
    setup_hyperparams, 
    get_model_pipeline, 
    get_augment_mode
)
from cp.cp_pipelines import run_wcp_trials


def wcp(hyperparams: DictConfig) -> None:
    
    _ = setup_hyperparams(hyperparams=hyperparams)

    # set seed
    pl.seed_everything(hyperparams.experiment.seed)

    # get model, callbacks, and image data
    model = get_model_pipeline(hyperparams)
    

    print(f"Running CIFAR{model.num_classes} with {hyperparams.cp.score_fn}")

    #Define transformation on calibration data. Calibration data aligns with underlying canonicalization model
    cal_augment_mode = get_augment_mode(hyperparams.checkpoint.checkpoint_name)
    hyperparams.dataset.augment = cal_augment_mode 
    
    image_data_cal = get_image_data(hyperparams.dataset)
    image_data_cal.setup(stage="test")
   

    #Define test data with possible shift away from calibration data (double shift setting)
    hyperparams.dataset.augment = hyperparams.cp.weighted_cp.shift_augment
    image_data_test = get_image_data(hyperparams.dataset)
    image_data_test.setup(stage="test")

    model.to("cuda")


    print("--------- SCP ---------")
    results = run_wcp_trials(
        model=model, 
        method="baseline",
        data_module=image_data_cal, 
        shifted_data_module=image_data_test,
        num_resamples=hyperparams.cp.num_resamples,
        score_fn=hyperparams.cp.score_fn,
        num_classes=model.num_classes,
        alpha=hyperparams.cp.alpha
    )

    print("--------- WCP ---------")
    results = run_wcp_trials(
        model=model, 
        method="weighted",
        data_module=image_data_cal, 
        shifted_data_module=image_data_test,
        num_resamples=hyperparams.cp.num_resamples,
        score_fn=hyperparams.cp.score_fn,
        num_classes=model.num_classes,
        alpha=hyperparams.cp.alpha,
        similarity_type=hyperparams.cp.similarity_type,
        w_lambda=hyperparams.cp.weight_lambda,
        w_pow=hyperparams.cp.weight_pow,
    )
    

# load the variables from .env file
load_envs()


@hydra.main(config_path=str("./configs/"), config_name="default")
def main(cfg: DictConfig) -> None:
    wcp(cfg)


if __name__ == "__main__":
    main()
