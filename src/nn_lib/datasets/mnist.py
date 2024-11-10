from torchvision.datasets import MNIST
from nn_lib.datasets.base import TorchvisionDataModuleBase, TorchvisionDatasetType


class MNISTDataModule(TorchvisionDataModuleBase):
    name = "mnist"
    _default_shape = (1, 28, 28)
    num_classes = 10
    type = TorchvisionDatasetType.IMAGE_CLASSIFICATION

    def train_data(self, transform=None, target_transform=None, transforms=None):
        if transforms is not None:
            raise ValueError("transforms argument must be None for MNIST")
        return MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

    def test_data(self, transform=None, target_transform=None, transforms=None):
        if transforms is not None:
            raise ValueError("transforms argument must be None for MNIST")
        return MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )
