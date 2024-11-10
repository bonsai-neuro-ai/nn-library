import torch
from torch.nn.utils.rnn import pad_sequence
from nn_lib.datasets.base import TorchvisionDataModuleBase, TorchvisionDatasetType
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2


class CocoDetectionDataModule(TorchvisionDataModuleBase):
    name = "coco"
    _default_shape = (3, 224, 224)
    num_classes = 21
    type = TorchvisionDatasetType.SEMANTIC_SEGMENTATION

    def train_data(self, transform=None, target_transform=None, transforms=None):
        ds = CocoDetection(
            root=self.data_dir / "images" / "train2017",
            annFile=self.data_dir / "annotations" / "instances_train2017.json",
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        return wrap_dataset_for_transforms_v2(ds, ("boxes", "labels", "masks"))

    def test_data(self, transform=None, target_transform=None, transforms=None):
        ds = CocoDetection(
            root=self.data_dir / "images" / "val2017",
            annFile=self.data_dir / "annotations" / "instances_val2017.json",
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        return wrap_dataset_for_transforms_v2(ds, ("boxes", "labels", "masks"))

    @staticmethod
    def collate_fn(batch):
        im_list, annot_list = zip(*batch)
        annot = {}
        for k in annot_list[0].keys():
            annot[k] = pad_sequence([a[k] for a in annot_list], batch_first=True)
        return torch.stack(im_list, 0), annot

    def train_dataloader(self, **kwargs):
        return super().train_dataloader(collate_fn=self.collate_fn, **kwargs)

    def val_dataloader(self, **kwargs):
        return super().val_dataloader(collate_fn=self.collate_fn, **kwargs)

    def test_dataloader(self, **kwargs):
        return super().test_dataloader(collate_fn=self.collate_fn, **kwargs)