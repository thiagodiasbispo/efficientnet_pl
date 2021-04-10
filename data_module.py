import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

import config as cf


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # download only
        datasets.ImageNet(
            "data", train=True, download=True, transform=transforms.ToTensor()
        )
        datasets.ImageNet(
            "data", train=False, download=True, transform=transforms.ToTensor()
        )

    def setup(self, stage):
        # transform
        transform = transforms.Compose([transforms.ToTensor()])
        training_dataset = datasets.ImageNet(
            "data", train=True, download=False, transform=transform
        )
        test_dataset = datasets.ImageNet(
            "data", train=False, download=False, transform=transform
        )

        # train/val split
        imagenet_train, imagenet_val = random_split(training_dataset, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = imagenet_train
        self.val_dataset = imagenet_val
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class CifarDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, dataset_name="cifar100"):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        if self.dataset_name == "cifar100":
            self.dataset = datasets.CIFAR100
        elif self.dataset_name == "cifar10":
            self.dataset = datasets.CIFAR10
        else:
            assert False

    def prepare_data(self):
        # download only
        self.dataset("data", train=True, download=True, transform=transforms.ToTensor())
        self.dataset(
            "data", train=False, download=True, transform=transforms.ToTensor()
        )

    def setup(self, stage):
        # transform
        train_transform = transforms.Compose(
            [
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    cf.mean[self.dataset_name], cf.std[self.dataset_name]
                ),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    cf.mean[self.dataset_name], cf.std[self.dataset_name]
                ),
            ]
        )

        if self.dataset_name == "cifar100":
            dataset = datasets.CIFAR100

        elif self.dataset_name == "cifar10":
            dataset = datasets.CIFAR10
        else:
            assert False

        training_dataset = dataset(
            "data", train=True, download=False, transform=train_transform
        )

        val_size = int(len(training_dataset) * 0.1)
        train_size = len(training_dataset) - val_size

        test = dataset("data", train=False, download=False, transform=test_transform)

        train, val = random_split(training_dataset, [train_size, val_size])

        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
