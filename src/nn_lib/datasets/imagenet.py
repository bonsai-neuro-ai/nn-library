import os
from torchvision.datasets import ImageFolder
from nn_lib.datasets.base import TorchvisionDataModuleBase
import yaml


class ImageNetDataModule(TorchvisionDataModuleBase):
    name = "imagenet"
    shape = (3, 224, 224)
    num_classes = 1000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.path.exists(os.path.join(self.data_dir, "class_ids.yaml")):
            with open(os.path.join(self.data_dir, "class_ids.yaml"), "r") as f:
                self.class_ids = yaml.safe_load(f)
        else:
            self.class_ids = None

    def train_data(self, transform=None):
        return ImageFolder(root=os.path.join(self.data_dir, "train"), transform=transform)

    def test_data(self, transform=None):
        return ImageFolder(root=os.path.join(self.data_dir, "val"), transform=transform)

    def class_id_to_label(self, class_id):
        if self.class_ids is None:
            return class_id
        return self.class_ids[class_id]