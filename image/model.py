# Adapted from https://github.com/arnab39/equiadapt/blob/main/examples/images/classification/model.py
# MIT License, (c) 2023  (c) 2023 Arnab Mondal

import pytorch_lightning as pl
import torch
from inference_utils import get_inference_method
from model_utils import get_dataset_specific_info, get_prediction_network
from omegaconf import DictConfig
from torch.optim.lr_scheduler import MultiStepLR

from common.utils import get_canonicalization_network, get_canonicalizer



def load_model_from_checkpoint(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    
    # Extract the hyperparameters from the checkpoint
    hyperparams = checkpoint['hyper_parameters']
    

    _, image_shape, num_classes = get_dataset_specific_info(
        hyperparams.dataset.dataset_name
    )

    # Construct the prediction network
    prediction_network = get_prediction_network(
        architecture=hyperparams['prediction']['prediction_network_architecture'],
        dataset_name=hyperparams['dataset']['dataset_name'],
        use_pretrained=hyperparams['prediction']['use_pretrained'],
        freeze_encoder=hyperparams['prediction']['freeze_pretrained_encoder'],
        input_shape=image_shape,  # This needs to be loaded or predefined
        num_classes=num_classes,  # This needs to be loaded or predefined
    )

    
    # Load the state dictionary into the prediction network
    prediction_network.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Construct the canonicalization network and canonicalizer
    canonicalization_network = get_canonicalization_network(
        hyperparams['canonicalization_type'],
        hyperparams['canonicalization'],
        image_shape  # This needs to be loaded or predefined
    )

    canonicalizer = get_canonicalizer(
        hyperparams['canonicalization_type'],
        canonicalization_network,
        hyperparams['canonicalization'],
        image_shape  # This needs to be loaded or predefined
    )
    
    # Assuming the canonicalizer also has state stored in the checkpoint
    # You might need to adjust this part depending on how the state is stored
    # This step is illustrative and might need modification
    canonicalizer.load_state_dict(checkpoint['state_dict'], strict=False)

    return prediction_network, canonicalizer

# define the LightningModule
class ImageClassifierPipeline(pl.LightningModule):
    def __init__(self, hyperparams: DictConfig):
        super().__init__()

        self.loss, self.image_shape, self.num_classes = get_dataset_specific_info(
            hyperparams.dataset.dataset_name
        )

        self.prediction_network = get_prediction_network(
            architecture=hyperparams.prediction.prediction_network_architecture,
            dataset_name=hyperparams.dataset.dataset_name,
            use_pretrained=hyperparams.prediction.use_pretrained,
            freeze_encoder=hyperparams.prediction.freeze_pretrained_encoder,
            input_shape=self.image_shape,
            num_classes=self.num_classes,
        )

        canonicalization_network = get_canonicalization_network(
            hyperparams.canonicalization_type,
            hyperparams.canonicalization,
            self.image_shape,
        )

        self.canonicalizer = get_canonicalizer(
            hyperparams.canonicalization_type,
            canonicalization_network,
            hyperparams.canonicalization,
            self.image_shape,
        )

        self.hyperparams = hyperparams

        self.inference_method = get_inference_method(
            self.canonicalizer,
            self.prediction_network,
            self.num_classes,
            hyperparams.experiment.inference,
            self.image_shape,
        )

        self.max_epochs = hyperparams.experiment.training.num_epochs

        self.save_hyperparameters()

    def forward(self, batch: torch.Tensor):
        x = batch

        batch_size, num_channels, height, width = x.shape

        # assert that the input is in the right shape
        assert (num_channels, height, width) == self.image_shape


        # canonicalize the input data
        # For the vanilla model, the canonicalization is the identity transformation
        x_canonicalized = self.canonicalizer(x)

        # Forward pass through the prediction network as you'll normally do
        logits = self.prediction_network(x_canonicalized)

        return logits


    def training_step(self, batch: torch.Tensor):
        x, y = batch[0], batch[1]
        batch_size, num_channels, height, width = x.shape

        # assert that the input is in the right shape
        assert (num_channels, height, width) == self.image_shape

        training_metrics = {}
        loss = 0.0

        # canonicalize the input data
        # For the vanilla model, the canonicalization is the identity transformation
        x_canonicalized = self.canonicalizer(x)

        # add group contrast loss while using optmization based canonicalization method
        if "opt" in self.hyperparams.canonicalization_type:
            group_contrast_loss = self.canonicalizer.get_optimization_specific_loss()
            loss += (
                group_contrast_loss
                * self.hyperparams.experiment.training.loss.group_contrast_weight
            )
            training_metrics.update(
                {"train/optimization_specific_loss": group_contrast_loss}
            )
            loss += (
                group_contrast_loss
                * self.hyperparams.experiment.training.loss.group_contrast_weight
            )
            training_metrics.update(
                {"train/optimization_specific_loss": group_contrast_loss}
            )

        # calculate the task loss which is the cross-entropy loss for classification
        if self.hyperparams.experiment.training.loss.task_weight:
            # Forward pass through the prediction network as you'll normally do
            logits = self.prediction_network(x_canonicalized)

            task_loss = self.loss(logits, y)
            loss += self.hyperparams.experiment.training.loss.task_weight * task_loss

            # Get the predictions and calculate the accuracy
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()

            training_metrics.update({"train/task_loss": task_loss, "train/acc": acc})
            training_metrics.update({"train/task_loss": task_loss, "train/acc": acc})

        # Add prior regularization loss if the prior weight is non-zero
        if self.hyperparams.experiment.training.loss.prior_weight:
            prior_loss = self.canonicalizer.get_prior_regularization_loss()
            loss += prior_loss * self.hyperparams.experiment.training.loss.prior_weight
            metric_identity = self.canonicalizer.get_identity_metric()
            training_metrics.update(
                {
                    "train/prior_loss": prior_loss,
                    "train/identity_metric": metric_identity,
                }
            )

        training_metrics.update(
            {
                "train/loss": loss,
            }
        )

        # Log the training metrics
        self.log_dict(training_metrics, prog_bar=True)
        assert not torch.isnan(loss), "Loss is NaN"

        return {"loss": loss, "acc": acc}

    def validation_step(self, batch: torch.Tensor):
        x, y = batch[0], batch[1]

        batch_size, num_channels, height, width = x.shape

        # assert that the input is in the right shape
        assert (num_channels, height, width) == self.image_shape

        validation_metrics = {}

        # canonicalize the input data
        # For the vanilla model, the canonicalization is the identity transformation
        x_canonicalized = self.canonicalizer(x)

        # Forward pass through the prediction network as you'll normally do
        logits = self.prediction_network(x_canonicalized)

        # Get the predictions and calculate the accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        # Log the identity metric if the prior weight is non-zero
        if self.hyperparams.experiment.training.loss.prior_weight:
            metric_identity = self.canonicalizer.get_identity_metric()
            validation_metrics.update({"train/identity_metric": metric_identity})

        # Logging to TensorBoard by default
        validation_metrics.update({"val/acc": acc})
        self.log_dict(
            {key: value.to(self.device) for key, value in validation_metrics.items()},
            prog_bar=True,
            sync_dist=True,
        )

        return {"acc": acc}

    def test_step(self, batch: torch.Tensor):
        x, y = batch[0], batch[1]
        batch_size, num_channels, height, width = x.shape

        # assert that the input is in the right shape
        assert (num_channels, height, width) == self.image_shape

        test_metrics = self.inference_method.get_inference_metrics(x, y)
        # Log the test metrics
        self.log_dict(
            {key: value.to(self.device) for key, value in test_metrics.items()},
            prog_bar=True,
            sync_dist=True,
        )

        return test_metrics

    def configure_optimizers(self):
        print(f"using AdamW optimizer")
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.prediction_network.parameters(),
                    "lr": self.hyperparams.experiment.training.prediction_lr,
                },
                {
                    "params": self.canonicalizer.parameters(),
                    "lr": self.hyperparams.experiment.training.canonicalization_lr,
                },
            ]
        )
        return optimizer
