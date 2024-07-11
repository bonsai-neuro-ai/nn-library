import torch
from torchvision.transforms import Normalize, Compose, ToTensor
from torch.utils.data import DataLoader, random_split
import lightning as lit
import os
from abc import ABCMeta, abstractmethod


class TorchvisionDataModuleBase(lit.LightningDataModule, metaclass=ABCMeta):
    def __init__(
        self,
        root_dir: str = "data",
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
        self.root_dir = root_dir
        self.train_ds_split, self.val_ds_split, self.test_ds = None, None, None

    @property
    def metadata(self):
        metadata_file = os.path.join(self.data_dir, "metadata.pkl")
        if not os.path.exists(metadata_file):
            self.prepare_data()
        return torch.load(metadata_file)

    @property
    def train_transform(self):
        return Compose([ToTensor(), Normalize(self.metadata["mean"], self.metadata["std"])])

    @property
    def test_transform(self):
        return Compose([ToTensor(), Normalize(self.metadata["mean"], self.metadata["std"])])

    @property
    def data_dir(self):
        return os.path.join(self.root_dir, self.name)

    @abstractmethod
    def train_data(self, transform=None):
        """Download the dataset if needed; must be implemented by subclass and return train
        dataset."""

    @abstractmethod
    def test_data(self, transform=None):
        """Download the dataset if needed; must be implemented by subclass and return train
        dataset."""

    def prepare_data(self) -> None:
        d = self.train_data(transform=ToTensor())
        _ = self.test_data(transform=ToTensor())

        metadata_file = os.path.join(self.data_dir, "metadata.pkl")

        if not os.path.exists(metadata_file):
            # Calculate mean and std of each channel of the dataset.
            im = next(iter(d))[0]
            num_channels = im.shape[0]
            moment1, moment2 = torch.zeros(num_channels), torch.zeros(num_channels)
            for i, (x, _) in enumerate(d):
                moment1 += x.mean([1, 2])
                moment2 += x.pow(2).mean([1, 2])
            mean = moment1 / len(d)
            std = (moment2 / len(d) - mean.pow(2)).sqrt()
            metadata = {"mean": mean, "std": std, "num_channels": num_channels, "n": len(d)}
            torch.save(metadata, metadata_file)

    def setup(self, stage: str):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit":
            data_full = self.train_data(transform=self.train_transform)
            n_train = int(len(data_full) * self.train_val_split)
            n_val = len(data_full) - n_train
            self.train_ds_split, self.val_ds_split = random_split(
                data_full, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed)
            )

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.test_ds = self.test_data(transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds_split, batch_size=self.bs, num_workers=self.nw, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds_split, batch_size=self.bs, num_workers=self.nw, persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.bs, num_workers=self.nw, persistent_workers=True
        )
