import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from hydra import initialize, compose
import pickle
import os
from train_utils import (
    get_model_pipeline,
    load_envs,
    setup_hyperparams,
    get_cifar_dataset,
)

# Calculate average color
def calculate_average_color(images):
    return images.mean(dim=[2, 3])


def quantize_color(color, num_bins):
    # Quantize color to the nearest bin
    return tuple((np.array(color) * (num_bins / 256)).astype(int))


def get_color_info(path: str) -> None:
    """
    Computes discretized average color bin for each CIFAR-10 test image
    and stores the bin indices as a tensor at the specified path.

    Args:
        path (str): Directory where 'color.pkl' will be saved.
    """
    # Transform to convert PIL images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load CIFAR-10 test set
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False)

    # Map from discretized RGB tuple to a bin index
    rgb_bins_2_idx = {
        (0, 0, 0): 0,
        (1, 0, 0): 1,
        (1, 1, 0): 2,
        (1, 1, 1): 3,
        (1, 0, 1): 4,
        (0, 1, 0): 5,
        (0, 1, 1): 6,
        (0, 0, 1): 7,
    }

    color_bins = []

    for images, _ in trainloader:
        # Compute average RGB color
        avg_color = calculate_average_color(images)

        # Discretize color (threshold at 0.5), map to bin index
        disc = (avg_color > 0.5).squeeze()
        bin_idx = rgb_bins_2_idx[(disc[0].item(), disc[1].item(), disc[2].item())]

        color_bins.append(bin_idx)

    # Convert list to tensor
    color_bins = torch.tensor(color_bins)

    # Ensure output directory exists
    os.makedirs(path, exist_ok=True)

    # Save bin indices to a file
    with open(path + '/color.pkl', 'wb') as handle:
        pickle.dump(color_bins, handle)


def _get_entropy(logits):
        
    # Calculate probabilities using softmax
    probs = torch.softmax(logits, dim=-1)
    # Calculate log probabilities using log_softmax
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

def _get_entropy_bin(entropies):
    entropy_bins = np.array([0.0, 1/10000, 1/1000, 1/100, 1/10, 1.0])
    indices = []
    for ent in entropies:
        idx = np.nonzero(ent.item() >= entropy_bins)
        indices.append(idx[0][-1])

    return indices

def _collect_entropy(path, model, device, dataloader):
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for examples in dataloader:
            batch_inputs, batch_labels = examples[0].to(device), examples[1].to(device)


            batch_logits = model(batch_inputs).detach()

            logits_list.append(batch_logits)
            labels_list.append(batch_labels)

    logits = torch.cat(logits_list).float().to(device)

    entropies = _get_entropy(logits)
    entropy_indices = torch.tensor(_get_entropy_bin(entropies), device=logits.device)
    os.makedirs(path, exist_ok=True)
    with open(path + 'entropy.pkl', 'wb') as handle:
        pickle.dump(entropy_indices, handle)
    

def get_label_info(path: str) -> None:
    """
    Stores CIFAR-10 test set labels as a tensor at the specified path.

    Args:
        path (str): Directory where the label file 'label.pkl' will be saved.
    """
    # Define transform to convert PIL images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load CIFAR-10 test set with transformation
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False)

    # Collect all labels
    labels = []
    for _, label in trainloader:
        labels.append(label)

    # Stack labels into a tensor
    labels = torch.tensor(labels)

    # Ensure output directory exists
    os.makedirs(path, exist_ok=True)

    # Save labels to a pickle file
    with open(path + 'label.pkl', 'wb') as handle:
        pickle.dump(labels, handle)


def get_entropy_info(path: str) -> None:
    """
    Runs a model on CIFAR-10 with small rotations and collects entropy of the softmax outputs
    as a proxy for image difficulty. Stores the result at the specified path.

    Args:
        path (str): Path to store the computed entropy information.
    """

    # Load environment variables and set up Hydra configuration overrides
    load_envs()

    overrides = [
        "canonicalization=identity",
        "dataset.dataset_name='cifar10'",
        "dataset.augment=small_rotations",
        "experiment.run_mode=test",
        "wandb.use_wandb=False",
        "checkpoint.checkpoint_path='./checkpoints'",
        "checkpoint.checkpoint_name='base'",
        "hydra.job.chdir=False",
        "dataset.cp_experiments.class_conditional_joint=0",
    ]

    # Initialize Hydra and compose the configuration
    with initialize(version_base=None, config_path="./configs"):
        hyperparams = compose(config_name="default", overrides=overrides)

    # Set up experiment parameters from the config
    _ = setup_hyperparams(hyperparams)

    # Sanity check to ensure test mode is active
    if hyperparams.experiment.run_mode != "test":
        raise Exception("run mode must be in test")

    # Seed for reproducibility
    pl.seed_everything(hyperparams.experiment.seed)
    
    # Load model and prepare data
    model = get_model_pipeline(hyperparams)

    image_data = get_cifar_dataset(
        hyperparams.dataset,
        version=model.num_classes,
        num_rotations=hyperparams.experiment.inference.num_rotations,
        partition=hyperparams.cp.mondrian.partitioning
    )
    image_data.setup(stage="test")

    # Compute and store entropy values from model predictions
    _collect_entropy(
        path=path,
        model=model,
        device=torch.device("cuda"),
        dataloader=image_data.test_dataloader()
    )