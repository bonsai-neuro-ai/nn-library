from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from nn_lib.datasets.base import TorchvisionDataModuleBase, TorchvisionDatasetType
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomRotation


class CIFAR10DataModule(TorchvisionDataModuleBase):
    name = "cifar10"
    _default_shape = (3, 32, 32)
    num_classes = 10
    type = TorchvisionDatasetType.IMAGE_CLASSIFICATION

    @property
    def train_transform(self):
        return Compose(
            [
                ToTensor(),
                RandomHorizontalFlip(),
                RandomRotation(10, fill=tuple(self.metadata["mean"].numpy())),
                Normalize(self.metadata["mean"], self.metadata["std"]),
            ]
        )

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
