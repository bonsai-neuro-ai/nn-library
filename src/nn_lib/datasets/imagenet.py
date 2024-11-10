import os
from torchvision.datasets import ImageFolder
from nn_lib.datasets.base import TorchvisionDataModuleBase, TorchvisionDatasetType
import yaml


class ImageNetDataModule(TorchvisionDataModuleBase):
    name = "imagenet"
    _default_shape = (3, 224, 224)
    num_classes = 1000
    type = TorchvisionDatasetType.IMAGE_CLASSIFICATION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.path.exists(os.path.join(self.data_dir, "class_ids.yaml")):
            with open(os.path.join(self.data_dir, "class_ids.yaml"), "r") as f:
                self.class_ids = yaml.safe_load(f)
        else:
            self.class_ids = None

    def train_data(self, transform=None, target_transform=None, transforms=None):
        if transforms is not None:
            raise ValueError("transforms argument must be None for ImageNet")
        return ImageFolder(
            root=os.path.join(self.data_dir, "train"),
            transform=transform,
            target_transform=target_transform,
        )

    def test_data(self, transform=None, target_transform=None, transforms=None):
        if transforms is not None:
            raise ValueError("transforms argument must be None for ImageNet")
        return ImageFolder(
            root=os.path.join(self.data_dir, "val"),
            transform=transform,
            target_transform=target_transform,
        )

    def class_id_to_label(self, class_id):
        if self.class_ids is None:
            return class_id
        return self.class_ids[class_id]
