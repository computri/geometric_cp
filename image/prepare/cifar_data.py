# Adapted from https://github.com/arnab39/equiadapt/blob/main/examples/images/classification/prepare/cifar_data.py
# MIT License, (c) 2023  (c) 2023 Arnab Mondal

import os
import pickle

import numpy as np
from scipy.stats import vonmises
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100



partition_2_file = {
    "color": "color_bins.pkl",
    "entropy": "entropy_indices.pkl"
}


class CustomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = float(np.random.choice(self.angles, p=None))

        return transforms.functional.rotate(x, angle)
    
class ContinuousRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, num_rotations=4, kappa=1.0):
        # Parameters
        kappa = kappa  # High concentration for a sharp peak
        distance = np.pi / int(num_rotations/2)
        # offset = np.pi / 4
        # offset = 0.0
        # means = [0 - offset, distance - offset, 2*distance - offset, 3*distance - offset]  # Equidistant points on the circle
        # weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights for simplicity

        means = [i*distance for i in range(num_rotations)]
        weights = [1/num_rotations for _ in range(num_rotations)]

        # Generate angles for evaluation
        angles = np.linspace(0, 2*np.pi, 200)

        # Compute the mixed probability density
        pdf_values = np.zeros_like(angles)
        for mu, weight in zip(means, weights):
            dist = vonmises(kappa, loc=mu)
            pdf_values += dist.pdf(angles) * weight

        self.angles = np.linspace(0, 360, 200)

        # pdf_values += 0.1
        pdf_values /= pdf_values.sum()
        self.p = pdf_values


    def __call__(self, x):
        angle = float(np.random.choice(self.angles, p=self.p))

        return transforms.functional.rotate(x, angle)
   

class AugCIFAR100(CIFAR100):
    def __init__(self, angles, partition_distributions, transform, **kwargs):
        fixed_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
            ]
        )
        super().__init__(**kwargs, transform=fixed_transform, target_transform=None)

        self.partition_distributions = partition_distributions

        self.angles = angles
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            ),
        ])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = super().__getitem__(index)

        angle = np.random.choice(self.angles, p=self.partition_distributions[target])

        img = transforms.functional.rotate(img, angle)

        if self.post_transform is not None:
            img = self.post_transform(img)

        return img, target#, angle

class AugCIFAR10(CIFAR10):
    def __init__(self, angles, partition_distributions, transform, partition='label', **kwargs):
        fixed_transform = transforms.Compose(
            [
                # transforms.RandomCrop(32, padding=4),
                transforms.Resize(224),
                # transforms.RandomHorizontalFlip(),
            ]
        )
        super().__init__(**kwargs, transform=fixed_transform, target_transform=None)

        self.partition_distributions = partition_distributions
        self.partition = partition
        self.angles = angles
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            ),
        ])
        if (partition != "label") and (partition != "none"):
            with open(f'./temp_data/partition_indices/{partition_2_file[self.partition]}', 'rb') as handle:
                self.partition_info = pickle.load(handle).to("cpu")


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = super().__getitem__(index)

        if self.partition == "label" or self.partition == "none":
            target_partition_info = target
            angle = float(np.random.choice(self.angles, p=self.partition_distributions[target]))
        else:
            
            target_partition_info = self.partition_info[index].item()
            angle = float(np.random.choice(self.angles, p=self.partition_distributions[self.partition_info[index].item()]))

        img = transforms.functional.rotate(img, angle)

        if self.post_transform is not None:
            img = self.post_transform(img)

        return img, target, target_partition_info
    

class ClassConditionalCIFARDataModule(pl.LightningDataModule):
    def __init__(self, hyperparams, partition_distributions, partition="label", version=10, num_rotations=8, download=False):
        super().__init__()
        self.data_path = hyperparams.data_path
        self.hyperparams = hyperparams

        self.partition_distributions = partition_distributions

        self.partition = partition

        if version == 10:
            self.cifar_version = AugCIFAR10 
        elif version == 100:
            self.cifar_version = AugCIFAR100

        if num_rotations == 8:
            self.angles = [0, 45, 90, 135, 180, 225, 270, 315]
            
        elif num_rotations == 4:
            self.angles = [0, 90, 180, 270]
            # all augmentations
            

        os.makedirs(self.data_path, exist_ok=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Good strategy for splitting data
            # cifar_full = CIFAR10(self.data_path, train=True, transform=self.transform, download=True)
            # self.train_dataset, self.valid_dataset = random_split(cifar_full, [45000, 5000])
            # print('Train dataset size: ', len(self.train_dataset))
            # print('Valid dataset size: ', len(self.valid_dataset))
            # Not a good strategy for splitting data but most papers use this
            self.train_dataset = self.cifar_version(
                root=self.data_path,
                angles=self.angles, 
                partition_distributions=self.partition_distributions, 
                train=True,
                transform=None,
                partition=self.partition,
                download=True,
            )
            self.valid_dataset = self.cifar_version(
                root=self.data_path,
                angles=self.angles, 
                partition_distributions=self.partition_distributions, 
                train=False,
                transform=None,
                partition=self.partition,
                download=True,
            )
        if stage == "test":
            test_dataset = self.cifar_version(
                root=self.data_path,
                angles=self.angles, 
                partition_distributions=self.partition_distributions, 
                train=False,
                transform=None,
                partition=self.partition,
                download=True,
            )
            self.test_dataset = test_dataset
            print("Test dataset size: ", len(self.test_dataset))



    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            self.hyperparams.batch_size,
            shuffle=True,
            num_workers=self.hyperparams.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            self.hyperparams.batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            self.hyperparams.batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
        )
        return test_loader


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, hyperparams, download=False):
        super().__init__()
        self.data_path = hyperparams.data_path
        self.hyperparams = hyperparams
        if hyperparams.augment == "small_rotations":
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )
        elif hyperparams.augment == "so2":
            # all augmentations
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    # CustomRotationTransform([0, 45, 90, 135, 180, 225, 270, 315]),
                    transforms.RandomRotation(180),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )
        elif hyperparams.augment == "autoaugment":
            # autoaugment
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.AutoAugment(
                        policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )

        elif hyperparams.augment == "c8":
            # all augmentations
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    CustomRotationTransform([0, 45, 90, 135, 180, 225, 270, 315]),
                    transforms.RandomRotation(180),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )
        elif hyperparams.augment == "c4":
            # all augmentations
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    CustomRotationTransform([0, 90, 180, 270]),
                    # transforms.RandomRotation(180),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )
        else:
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )


        test_transform = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
        
        if hyperparams.augment == "so2":
            test_transform.insert(1, transforms.RandomRotation(180))
        elif hyperparams.augment == "c8":
            # test_transform.insert(1, transforms.RandomRotation(180))
            test_transform.insert(1, CustomRotationTransform([0, 45, 90, 135, 180, 225, 270, 315]))
        elif hyperparams.augment == "c4":
            test_transform.insert(1, CustomRotationTransform([0, 90, 180, 270]))
        elif hyperparams.augment == "continuous_c4":
            test_transform.insert(1, ContinuousRotationTransform(num_rotations=4, kappa=hyperparams.cp_experiments.rotation_distribution.kappa))
        elif hyperparams.augment == "continuous_c8":
            test_transform.insert(1, ContinuousRotationTransform(num_rotations=8, kappa=hyperparams.cp_experiments.rotation_distribution.kappa))

        self.test_transform = transforms.Compose(test_transform)
        os.makedirs(self.data_path, exist_ok=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Good strategy for splitting data
            # cifar_full = CIFAR10(self.data_path, train=True, transform=self.transform, download=True)
            # self.train_dataset, self.valid_dataset = random_split(cifar_full, [45000, 5000])
            # print('Train dataset size: ', len(self.train_dataset))
            # print('Valid dataset size: ', len(self.valid_dataset))
            # Not a good strategy for splitting data but most papers use this
            self.train_dataset = CIFAR10(
                self.data_path,
                train=True,
                transform=self.train_transform,
                download=True,
            )
            self.valid_dataset = CIFAR10(
                self.data_path,
                train=False,
                transform=self.test_transform,
                download=True,
            )
        if stage == "test":
            self.test_dataset = CIFAR10(
                self.data_path,
                train=False,
                transform=self.test_transform,
                download=True,
            )
            print("Test dataset size: ", len(self.test_dataset))

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            self.hyperparams.batch_size,
            shuffle=True,
            num_workers=self.hyperparams.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            self.hyperparams.batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            self.hyperparams.batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
        )
        return test_loader


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, hyperparams, download=False):
        super().__init__()
        self.data_path = hyperparams.data_path
        self.hyperparams = hyperparams
        if hyperparams.augment == "small_rotations":
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )
        elif hyperparams.augment == "so2":
            # all augmentations
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    # CustomRotationTransform([0, 45, 90, 135, 180, 225, 270, 315]),
                    transforms.RandomRotation(180),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )
        elif hyperparams.augment == "c8":
            # all augmentations
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    CustomRotationTransform([0, 45, 90, 135, 180, 225, 270, 315]),
                    # transforms.RandomRotation(180),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )
        elif hyperparams.augment == "c4":
            # all augmentations
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    CustomRotationTransform([0, 90, 180, 270]),
                    # transforms.RandomRotation(180),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )
        else:
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),
                ]
            )


        test_transform = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
        

        if hyperparams.augment == "c8":

            test_transform.insert(1, CustomRotationTransform([0, 45, 90, 135, 180, 225, 270, 315]))
        elif hyperparams.augment == "c4":
            test_transform.insert(1, CustomRotationTransform([0, 90, 180, 270]))

        self.test_transform = transforms.Compose(test_transform)
        os.makedirs(self.data_path, exist_ok=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Good strategy for splitting data
            # cifar_full = CIFAR10(self.data_path, train=True, transform=self.transform, download=True)
            # self.train_dataset, self.valid_dataset = random_split(cifar_full, [45000, 5000])
            # print('Train dataset size: ', len(self.train_dataset))
            # print('Valid dataset size: ', len(self.valid_dataset))
            # Not a good strategy for splitting data but most papers use this
            self.train_dataset = CIFAR100(
                self.data_path,
                train=True,
                transform=self.train_transform,
                download=True,
            )
            self.valid_dataset = CIFAR100(
                self.data_path,
                train=False,
                transform=self.test_transform,
                download=True,
            )
        if stage == "test":
            self.test_dataset = CIFAR100(
                self.data_path,
                train=False,
                transform=self.test_transform,
                download=True,
            )
            print("Test dataset size: ", len(self.test_dataset))

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            self.hyperparams.batch_size,
            shuffle=True,
            num_workers=self.hyperparams.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            self.hyperparams.batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            self.hyperparams.batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
        )
        return test_loader
