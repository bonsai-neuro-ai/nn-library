import torch
from torch.utils.data import DataLoader, random_split
import lightning as lit
import os
from abc import ABCMeta, abstractmethod
from tqdm.auto import tqdm
from pathlib import Path
from nn_lib.datasets.transforms import get_default_transforms_v2
from nn_lib.datasets.enums import TorchvisionDatasetType
import torchvision.transforms.v2 as tv_transforms


class TorchvisionDataModuleBase(lit.LightningDataModule, metaclass=ABCMeta):
    __metafields__ = frozenset({"root_dir", "num_workers"})

    _default_shape: tuple[int, int, int] = None
    name: str = None
    type: TorchvisionDatasetType = None

    def __init__(
        self, root_dir: str | Path = "data", train_val_split: float = 11 / 12, seed: int = 8675309
    ):
        super().__init__()
        self.train_val_split = train_val_split
        self.seed = seed
        self.root_dir = Path(root_dir)
        self.train_ds_split = self.val_ds_split = self.test_ds = None
        self.train_transform = self.test_transform = None

    @property
    def shape(self):
        # The .shape property defaults to the cls._default_shape attribute unless the default
        # transform has been overridden. In that case, the shape is determined by the output of
        # the transform. Conversely, if the transform has not been set, the default transform
        # gets its shape from the cls._default_shape attribute. Basically, the shape attribute
        # should never be set directly, but rather by setting the default transform.
        if self.test_transform is not None:
            return self.test_transform(torch.zeros(1, *self._default_shape)).shape[1:]
        return self._default_shape

    @property
    def num_classes(self):
        """Return the number of classes in the dataset (classification tasks only)."""
        # TODO - restructure so not all datasets need this property
        return None

    @property
    def metadata(self):
        metadata_file = os.path.join(self.data_dir, "metadata.pkl")
        if not os.path.exists(metadata_file):
            self.prepare_data()
        return torch.load(metadata_file, weights_only=False, map_location="cpu")

    @property
    def data_dir(self):
        return self.root_dir / self.name

    @abstractmethod
    def train_data(self, transform=None, target_transform=None, transforms=None):
        """Download the dataset if needed; must be implemented by subclass and return train
        dataset."""

    @abstractmethod
    def test_data(self, transform=None, target_transform=None, transforms=None):
        """Download the dataset if needed; must be implemented by subclass and return train
        dataset."""

    @property
    def requires_target_transform(self):
        return self.type in (
            TorchvisionDatasetType.SEMANTIC_SEGMENTATION,
            TorchvisionDatasetType.OBJECT_DETECTION,
            TorchvisionDatasetType.OPTICAL_FLOW,
        )

    def prepare_data(self) -> None:
        # In the absence of transforms specified by the caller, we will do a pass over the data
        # to calculate statistics for normalization; the metadata calculated here will be used by
        # get_default_transforms_v2 to normalize the data.
        if self.train_transform is None and self.test_transform is None:
            metadata_file = os.path.join(self.data_dir, "metadata.pkl")
            if not os.path.exists(metadata_file):
                # Calculate mean and std of each channel of the dataset.
                d = self.train_data(transform=tv_transforms.ToTensor())
                im = next(iter(d))[0]
                num_channels = im.shape[0]
                moment1, moment2 = torch.zeros(num_channels), torch.zeros(num_channels)
                for i, (x, _) in tqdm(enumerate(d), total=len(d), desc="One-time dataset stats"):
                    moment1 += x.mean([1, 2])
                    moment2 += x.pow(2).mean([1, 2])
                mean = moment1 / len(d)
                std = (moment2 / len(d) - mean.pow(2)).sqrt()
                metadata = {"mean": mean, "std": std, "num_channels": num_channels, "n": len(d)}
                torch.save(metadata, metadata_file)

        if self.train_transform is None:
            self.train_transform = get_default_transforms_v2(self)

        if self.test_transform is None:
            self.test_transform = get_default_transforms_v2(self)

    def setup(self, stage: str):
        # Assign Train/val split(s) for use in Dataloaders
        if stage in ("fit", "val"):
            # Note that any changes to the train_transform or test_transform properties *after*
            # the first call to setup() will not be reflected in the train/val dataloaders.
            if self.requires_target_transform:
                data_full = self.train_data(transforms=self.train_transform)
            else:
                data_full = self.train_data(transform=self.train_transform)
            n_train = int(len(data_full) * self.train_val_split)
            n_val = len(data_full) - n_train
            self.train_ds_split, self.val_ds_split = random_split(
                data_full, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed)
            )

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            if self.requires_target_transform:
                self.test_ds = self.test_data(transforms=self.test_transform)
            else:
                self.test_ds = self.test_data(transform=self.test_transform)

    def train_dataloader(self, batch_size: int = 100, num_workers: int = 4, **kwargs):
        kwargs["pin_memory"] = kwargs.get("pin_memory", num_workers > 0)
        return DataLoader(
            self.train_ds_split,
            batch_size=batch_size,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            **kwargs,
        )

    def val_dataloader(self, batch_size: int = 100, num_workers: int = 4, **kwargs):
        kwargs["pin_memory"] = kwargs.get("pin_memory", num_workers > 0)
        return DataLoader(
            self.val_ds_split,
            batch_size=batch_size,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            **kwargs,
        )

    def test_dataloader(self, batch_size: int = 100, num_workers: int = 4, **kwargs):
        kwargs["pin_memory"] = kwargs.get("pin_memory", num_workers > 0)
        return DataLoader(
            self.test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            **kwargs,
        )
