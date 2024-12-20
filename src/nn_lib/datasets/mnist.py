from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from nn_lib.datasets.base import TorchvisionDataModuleBase, TorchvisionDatasetType


class MNISTDataModule(TorchvisionDataModuleBase):
    name = "mnist"
    _default_shape = (1, 28, 28)
    num_classes = 10
    type = TorchvisionDatasetType.IMAGE_CLASSIFICATION

    @property
    def transform(self):
        return ToTensor()

    def train_data(self, transform=None):
        return MNIST(self.data_dir, train=True, download=True, transform=transform)

    def test_data(self, transform=None):
        return MNIST(self.data_dir, train=False, download=True, transform=transform)
