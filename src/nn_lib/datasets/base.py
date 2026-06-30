from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
import os
from abc import ABCMeta, abstractmethod
from tqdm.auto import tqdm
from pathlib import Path
from nn_lib.datasets.transforms import get_default_transforms_v2
from nn_lib.datasets.enums import TorchvisionDatasetType
import torchvision.transforms.v2 as tv_transforms


class TorchvisionDataModuleBase(metaclass=ABCMeta):
    """
    Lightning-style DataModule base class wrapping a torchvision dataset, providing
    train/val/test splits and dataloaders with sensible defaults (default normalization
    transforms, reproducible shuffling/splitting via `seed`).

    Subclasses must set the `name`, `_default_shape`, and `type` class attributes, and implement
    `train_data`/`test_data` to construct the underlying torchvision `Dataset`.

    Note: subclasses are expected to call torchvision dataset constructors with `download=False`.
    This library does not download datasets automatically; data must already exist under
    `root_dir/name` (see `data_dir`) before use.
    """

    _default_shape: tuple[int, int, int] = None
    name: str = None
    type: TorchvisionDatasetType = None

    def __init__(
        self,
        root_dir: Optional[str | Path] = None,
        train_val_split: float = 11 / 12,
        seed: int = 8675309,
    ):
        """
        :param root_dir: directory containing dataset subfolders, one per dataset `name`. If
            None, falls back to the `DATA_ROOT` environment variable, then to "./data".
        :param train_val_split: fraction of the training data to use for the train split; the
            remainder becomes the validation split (see `setup`).
        :param seed: random seed used for the train/val split and for dataloader shuffling, to
            keep splits reproducible across runs.
        """
        super().__init__()
        self.train_val_split = train_val_split
        self.seed = seed
        # root_dir is configured with varying degrees of precedence. If passed in the
        # constructor, it will be used. Otherwise, we'll look for an environment variable.
        # Finally, if neither of those are set, we'll default to "./data"
        if root_dir is not None:
            self.root_dir = Path(root_dir)
        else:
            self.root_dir = Path(os.environ.get("DATA_ROOT", "./data"))
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
        """Directory this dataset's files are expected to live in: `root_dir / name`."""
        return self.root_dir / self.name

    @abstractmethod
    def train_data(self, transform=None, target_transform=None, transforms=None):
        """Construct and return the underlying torchvision train dataset. Must be implemented by
        subclass. Data is expected to already exist on disk (not downloaded automatically)."""

    @abstractmethod
    def test_data(self, transform=None, target_transform=None, transforms=None):
        """Construct and return the underlying torchvision test dataset. Must be implemented by
        subclass. Data is expected to already exist on disk (not downloaded automatically)."""

    @property
    def requires_target_transform(self):
        """True if this dataset's task type needs a joint input/target transform (`transforms=`)
        instead of separate `transform`/`target_transform` (e.g. segmentation, detection, flow,
        where the transform must be applied consistently to both image and target)."""
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
                d = self.train_data(
                    transform=tv_transforms.Compose(
                        [tv_transforms.ToImage(), tv_transforms.ToDtype(torch.float32, scale=True)]
                    )
                )
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
            shuffle=kwargs.pop("shuffle", self.seed is not None),
            **kwargs,
        )

    def val_dataloader(self, batch_size: int = 100, num_workers: int = 4, **kwargs):
        kwargs["pin_memory"] = kwargs.get("pin_memory", num_workers > 0)
        return DataLoader(
            self.val_ds_split,
            batch_size=batch_size,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            shuffle=kwargs.pop("shuffle", self.seed is not None),
            **kwargs,
        )

    def test_dataloader(self, batch_size: int = 100, num_workers: int = 4, **kwargs):
        kwargs["pin_memory"] = kwargs.get("pin_memory", num_workers > 0)
        return DataLoader(
            self.test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(self.seed),
            shuffle=kwargs.pop("shuffle", self.seed is not None),
            **kwargs,
        )
