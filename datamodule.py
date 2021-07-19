import os
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import CIFAR10, ImageFolder
import random

class RandomRotationDiscrete():
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class PairedImageFolder():
    def __init__(self, data_path, target_path, data_transform=None, target_transform=None):
        self.data_paths = [os.path.join(data_path, f) for f in os.listdir(data_path)]
        self.target_paths = [os.path.join(target_path, f) for f in os.listdir(target_path)]

        self.data_transform = data_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = Image.open(self.data_paths[index])
        if self.data_transform:
            x = self.data_transform(x)

        y = Image.open(self.target_paths[index])
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.data_paths)

class BaseDatamodule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.devset, batch_size=self.batch_size, shuffle=False, num_workers=2)

class CifarDatamodule(BaseDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_transform = transforms.Compose(
                    [
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(),
                    #RandomRotationDiscrete([0, 90, 180, 270]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        self.dev_transform = transforms.Compose(
                    [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

    def setup(self, stage=None):
        self.trainset = CIFAR10(root='./data', train=True,
                download=True, transform=self.train_transform)
        self.devset = CIFAR10(root='./data', train=False,
                download=True, transform=self.dev_transform)

class StarDatamodule(BaseDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_path = 'data/StarsGalaxies/train'
        self.dev_path = 'data/StarsGalaxies/dev'

        self.train_transform = transforms.Compose(
                    [
                    transforms.Grayscale(),
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(),
                    #RandomRotationDiscrete([0, 90, 180, 270]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                    ])
        self.dev_transform = transforms.Compose(
                    [
                    transforms.Grayscale(),
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                    ])

    def setup(self, stage=None):
        self.trainset = ImageFolder(self.train_path, transform=self.train_transform)
        self.devset = ImageFolder(self.dev_path, transform=self.dev_transform)

class CarsDatamodule(BaseDatamodule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_images_path = 'data/cars/train/images'
        self.train_masks_path = 'data/cars/train/masks'
        self.dev_images_path = 'data/cars/dev/images'
        self.dev_masks_path = 'data/cars/dev/masks'

        self.train_images_transform = transforms.Compose(
                    [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        self.train_masks_transform = transforms.Compose(
                    [
                    transforms.Grayscale(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    ])

        self.dev_images_transform = transforms.Compose(
                    [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        self.dev_masks_transform = transforms.Compose(
                    [
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    ])

    def setup(self, stage=None):
        self.trainset = PairedImageFolder(self.train_images_path, self.train_masks_path,
                                          data_transform=self.train_images_transform,
                                          target_transform=self.train_masks_transform)

        self.devset = PairedImageFolder(self.dev_images_path, self.dev_masks_path,
                                        data_transform=self.dev_images_transform,
                                        target_transform=self.dev_masks_transform)
