from torchvision.datasets import CIFAR10, CIFAR100
from nn_lib.datasets.base import TorchvisionDataModuleBase, TorchvisionDatasetType


class CIFAR10DataModule(TorchvisionDataModuleBase):
    name = "cifar10"
    _default_shape = (3, 32, 32)
    num_classes = 10
    type = TorchvisionDatasetType.IMAGE_CLASSIFICATION

    def train_data(self, transform=None, target_transform=None, transforms=None):
        if transforms is not None:
            raise ValueError("transforms must be None for CIFAR10")
        return CIFAR10(
            self.data_dir,
            train=True,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

    def test_data(self, transform=None, target_transform=None, transforms=None):
        if transforms is not None:
            raise ValueError("transforms must be None for CIFAR10")
        return CIFAR10(
            self.data_dir,
            train=False,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )


class CIFAR100DataModule(TorchvisionDataModuleBase):
    name = "cifar100"
    _default_shape = (3, 32, 32)
    num_classes = 100
    type = TorchvisionDatasetType.IMAGE_CLASSIFICATION

    def train_data(self, transform=None, target_transform=None, transforms=None):
        if transforms is not None:
            raise ValueError("transforms must be None for CIFAR100")
        return CIFAR100(
            self.data_dir,
            train=True,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

    def test_data(self, transform=None, target_transform=None, transforms=None):
        if transforms is not None:
            raise ValueError("transforms must be None for CIFAR100")
        return CIFAR100(
            self.data_dir,
            train=False,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )
