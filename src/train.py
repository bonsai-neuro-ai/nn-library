from torch import nn
import lightning as lit
from nn_lib.datasets import CIFAR10DataModule, CIFAR100DataModule
from nn_lib.models import CIFARResNet
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy
from typing import Union, Optional
from datetime import timedelta
from dataclasses import dataclass
import jsonargparse


class LitCIFARResNet(lit.LightningModule):
    # TODO - refactor this to the nn_lib.models module as a generic base for classification problems.
    def __init__(self, depth: int, width: int, num_classes: int = 10):
        super().__init__()
        self.model = CIFARResNet(f"cifar{num_classes}_{depth}_{width}")
        self.loss = nn.CrossEntropyLoss()
        self.metrics = {
            "acc": Accuracy("multiclass", num_classes=self.model.num_classes),
        }

    def forward(self, x):
        return self.model(x)["fc"]

    def to(self, device):
        out = super().to(device)
        for metric in self.metrics.values():
            metric.to(device)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        for name, metric in self.metrics.items():
            self.log(f"train_{name}", metric(y_hat, y))
        return loss

    def on_train_epoch_start(self):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        for name, metric in self.metrics.items():
            self.log(f"val_{name}", metric(y_hat, y))
        return loss

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=1e-3)
        sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-6)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

    def configure_callbacks(self):
        # Note: important that EarlyStopping patience is larger than ReduceLROnPlateau patience
        # so that the model has a chance to recover from a learning rate drop
        return [
            ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", save_last=True),
            EarlyStopping(monitor="val_loss", patience=10),
        ]


@dataclass
class TrainerConfig:
    """Configuration for the Trainer class; just the subset of Trainer arguments that significantly
    affect the training process and must be logged in a config file.
    """

    # TODO - write helpers to annotate some fields as hyperparameters that affect the model and
    #  others as 'local config'. The accelerator/strategy/devices fields are 'local config', e.g.
    accelerator: str = "auto"
    strategy: str = "auto"
    devices: Union[list[int], str, int] = "auto"
    fast_dev_run: Optional[Union[int, bool]] = None
    # --- above: local config, below: hyperparameters that matter ---
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    max_time: Optional[Union[str, timedelta, dict[str, int]]] = None
    limit_train_batches: Optional[Union[int, float]] = None
    limit_val_batches: Optional[Union[int, float]] = None
    limit_test_batches: Optional[Union[int, float]] = None
    limit_predict_batches: Optional[Union[int, float]] = None
    overfit_batches: Union[int, float] = 0.0
    val_check_interval: Optional[Union[int, float]] = None
    check_val_every_n_epoch: Optional[int] = 1
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[Union[int, float]] = None
    gradient_clip_algorithm: Optional[str] = None


@dataclass
class EnvConfig:
    """Local environment configuration."""

    mlflow_tracking_uri: Optional[str] = None


@dataclass
class MainConfig:
    """Dataclass specifying CLI args to the main() function."""

    expt_name: str
    model: LitCIFARResNet
    data: Union[CIFAR10DataModule, CIFAR100DataModule]
    trainer: TrainerConfig
    env: EnvConfig


def main(args: jsonargparse.Namespace):
    # Log using MLFlow
    logger = MLFlowLogger(experiment_name=args.expt_name, tracking_uri=args.env.mlflow_tracking_uri)

    # TODO – break early if this run is already complete
    # TODO - verify resume from checkpoint is working with this Trainer

    # Remove the config arguments from the args namespace; they just clutter the parameters log.
    if hasattr(args, "config"):
        delattr(args, "config")
    if hasattr(args, "__default_config__"):
        delattr(args, "__default_config__")

    # Save run metadata to the logger -- using the fact that the log_hyperparams method can take
    # a namespace object directly, and we have a namespace object for MainConfig.
    logger.log_hyperparams(args)

    # Call instantiate_classes to recursively instantiate all classes in the config. For example,
    # args.data will be a Namespace but args_with_instances.data will be an instance of a
    # LightningDataModule class.
    args_with_instances = parser.instantiate_classes(args)

    # Create the MainConfig object from the instantiated arguments.
    cfg = MainConfig(**args_with_instances.as_dict())

    # Create the trainer object using our custom logger and set the remaining arguments from the
    # TrainerConfig.
    trainer = lit.Trainer(logger=logger, **cfg.trainer.__dict__)
    trainer.fit(cfg.model, cfg.data, ckpt_path="last")


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser(default_config_files=["configs/local_config.yaml"])
    parser.add_class_arguments(MainConfig)
    parser.add_argument("--config", action="config")
    args = parser.parse_args()

    main(args)
