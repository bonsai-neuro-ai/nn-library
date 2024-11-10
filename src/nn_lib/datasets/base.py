import torch
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms._presets import (
    ImageClassification,
    ObjectDetection,
    OpticalFlow,
    SemanticSegmentation,
    VideoClassification,
)
from torch.utils.data import DataLoader, random_split
import lightning as lit
import os
from abc import ABCMeta, abstractmethod
from tqdm.auto import tqdm
from typing import Union, assert_never
from enum import Enum, auto
from pathlib import Path
import warnings


TransformType = Union[
    ImageClassification,
    ObjectDetection,
    OpticalFlow,
    SemanticSegmentation,
    VideoClassification,
]


class TorchvisionDatasetType(Enum):
    OBJECT_DETECTION = auto()
    IMAGE_CLASSIFICATION = auto()
    VIDEO_CLASSIFICATION = auto()
    SEMANTIC_SEGMENTATION = auto()
    OPTICAL_FLOW = auto()


class TorchvisionDataModuleBase(lit.LightningDataModule, metaclass=ABCMeta):
    __metafields__ = frozenset({"root_dir", "num_workers"})

    _default_shape: tuple[int, int, int] = None
    name: str = None
    type: TorchvisionDatasetType = None


    def __init__(
        self,
        root_dir: str | Path = "data",
        train_val_split: float = 11 / 12,
        seed: int = 8675309,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_val_split = train_val_split
        self.seed = seed
        self.bs = batch_size
        self.nw = num_workers
        self.root_dir = Path(root_dir)
        self.train_ds_split, self.val_ds_split, self.test_ds = None, None, None
        self._override_default_transform = None

    @property
    def shape(self):
        # The .shape property defaults to the cls._default_shape attribute unless the default
        # transform has been overridden. In that case, the shape is determined by the output of
        # the transform. Conversely, if the transform has not been set, the default transform
        # gets its shape from the cls._default_shape attribute. Basically, the shape attribute
        # should never be set directly, but rather by setting the default transform.
        if self._override_default_transform is not None:
            return self._override_default_transform(torch.zeros(1, *self._default_shape)).shape[1:]
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
        return torch.load(metadata_file)

    @property
    def default_transform(self) -> TransformType:
        if self._override_default_transform is not None:
            return self._override_default_transform
        match self.type:
            case TorchvisionDatasetType.IMAGE_CLASSIFICATION:
                return ImageClassification(
                    crop_size=self._default_shape[1],
                    resize_size=self._default_shape[1],
                    mean=self.metadata["mean"],
                    std=self.metadata["std"],
                )
            case TorchvisionDatasetType.OBJECT_DETECTION:
                raise NotImplementedError()  # TODO: TorchvisionDatasetType.OBJECT_DETECTION
            case TorchvisionDatasetType.VIDEO_CLASSIFICATION:
                raise NotImplementedError()  # TODO: TorchvisionDatasetType.VIDEO_CLASSIFICATION
            case TorchvisionDatasetType.SEMANTIC_SEGMENTATION:
                return SemanticSegmentation(
                    resize_size=self._default_shape[1],
                    mean=self.metadata["mean"],
                    std=self.metadata["std"],
                )
            case TorchvisionDatasetType.OPTICAL_FLOW:
                raise NotImplementedError()  # TODO: TorchvisionDatasetType.OPTICAL_FLOW
            case _:
                assert_never(self.type)

    @default_transform.setter
    def default_transform(self, transform: TransformType):
        match self.type:
            case TorchvisionDatasetType.IMAGE_CLASSIFICATION:
                expected_type = ImageClassification
            case TorchvisionDatasetType.OBJECT_DETECTION:
                expected_type = ObjectDetection
            case TorchvisionDatasetType.VIDEO_CLASSIFICATION:
                expected_type = VideoClassification
            case TorchvisionDatasetType.SEMANTIC_SEGMENTATION:
                expected_type = SemanticSegmentation
            case TorchvisionDatasetType.OPTICAL_FLOW:
                expected_type = OpticalFlow
            case _:
                assert_never(self.type)
        if not isinstance(transform, expected_type):
            warnings.warn(f"Expected transform of type {expected_type}, got {type(transform)}")
        self._override_default_transform = transform

    @property
    def train_transform(self):
        return Compose([ToTensor(), self.default_transform])

    @property
    def test_transform(self):
        return Compose([ToTensor(), self.default_transform])

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

    def prepare_data(self) -> None:
        # TODO - perhaps metadata should depend on transforms. e.g. we may want an input transform
        #  that does its own color equalization.
        d = self.train_data(transform=ToTensor())
        _ = self.test_data(transform=ToTensor())

        metadata_file = os.path.join(self.data_dir, "metadata.pkl")

        if not os.path.exists(metadata_file):
            # Calculate mean and std of each channel of the dataset.
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

    def setup(self, stage: str):
        # Assign Train/val split(s) for use in Dataloaders
        if stage in ("fit", "val"):
            data_full = self.train_data(transform=self.train_transform)
            n_train = int(len(data_full) * self.train_val_split)
            n_val = len(data_full) - n_train
            self.train_ds_split, self.val_ds_split = random_split(
                data_full, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed)
            )

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.test_ds = self.test_data(transform=self.test_transform)

    def train_dataloader(self, **kwargs):
        return DataLoader(
            self.train_ds_split,
            batch_size=self.bs,
            num_workers=self.nw,
            persistent_workers=self.nw > 0,
            generator=torch.Generator().manual_seed(self.seed),
            **kwargs,
        )

    def val_dataloader(self, **kwargs):
        return DataLoader(
            self.val_ds_split,
            batch_size=self.bs,
            num_workers=self.nw,
            persistent_workers=self.nw > 0,
            generator=torch.Generator().manual_seed(self.seed),
            **kwargs,
        )

    def test_dataloader(self, **kwargs):
        return DataLoader(
            self.test_ds,
            batch_size=self.bs,
            num_workers=self.nw,
            persistent_workers=self.nw > 0,
            generator=torch.Generator().manual_seed(self.seed),
            **kwargs,
        )
