from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from nn_lib.datasets.base import TorchvisionDataModuleBase, TorchvisionDatasetType
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomRotation


class CIFAR10DataModule(TorchvisionDataModuleBase):
    name = "cifar10"
    shape = (3, 32, 32)
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

    def train_data(self, transform=None):
        return CIFAR10(self.data_dir, train=True, download=True, transform=transform)

    def test_data(self, transform=None):
        return CIFAR10(self.data_dir, train=False, download=True, transform=transform)


class CIFAR100DataModule(TorchvisionDataModuleBase):
    name = "cifar100"
    shape = (3, 32, 32)
    num_classes = 100

    def train_data(self, transform=None):
        return CIFAR100(self.data_dir, train=True, download=True, transform=transform)

    def test_data(self, transform=None):
        return CIFAR100(self.data_dir, train=False, download=True, transform=transform)
