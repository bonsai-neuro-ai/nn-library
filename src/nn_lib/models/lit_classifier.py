from typing import Union, Sequence, Any, TypeVar, Generic, Literal

import lightning as lit
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.tuner import Tuner
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

T = TypeVar("T")


class LitClassifier(lit.LightningModule, Generic[T]):
    """A simple classification wrapper for a nn.Module."""

    def __init__(
        self,
        model: T,
        num_classes: int,
        lr: float | Literal["auto"] = "auto",
        label_smoothing: float = 0.0,
        scheduler_patience: int = 3,
        stopping_patience: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model: T = model
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.metrics = {
            "acc": Accuracy("multiclass", num_classes=num_classes),
            "raw_ce": nn.CrossEntropyLoss(),
        }
        self.lr = lr
        self.num_classes = num_classes
        self.scheduler_patience = scheduler_patience
        self.stopping_patience = stopping_patience

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        out = super().to(device)
        for metric in self.metrics.values():
            metric.to(device)
        return out

    def on_fit_start(self) -> None:
        if self.lr == "auto":
            if self.trainer.num_devices > 1:
                raise ValueError("Tuning is not supported with multi-GPU training.")

            tuner = Tuner(self.trainer)
            tuner.lr_find(self, datamodule=self.trainer.datamodule)

            print("Suggested learning rate:", self.lr)

        self.logger.    log_hyperparams({"initial_lr": self.lr})

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        for name, metric in self.metrics.items():
            self.log(f"train_{name}", metric(y_hat, y))
        return loss

    def on_train_epoch_start(self):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True)

    def on_validation_start(self) -> None:
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)
        for name, metric in self.metrics.items():
            self.log(f"val_{name}", metric(y_hat, y), sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, sync_dist=True)
        for name, metric in self.metrics.items():
            self.log(f"test_{name}", metric(y_hat, y), sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr)
        if self.scheduler_patience <= 0:
            return opt

        sched = ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.scheduler_patience, min_lr=self.lr / 32
        )
        if isinstance(self.trainer.val_check_interval, float):
            raise NotImplementedError(
                "Not yet handling val_check_interval as a float for increased LR scheduler updates"
            )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": self.trainer.val_check_interval,
                "monitor": "val_loss",
            },
        }

    def configure_callbacks(self) -> Union[Sequence[lit.Callback], lit.Callback]:
        # Note: important that EarlyStopping patience is larger than ReduceLROnPlateau patience
        # so that the model has a chance to recover from a learning rate drop
        # TODO - verify that early stopping (and reduceLR) work as expected when validating more
        #  than once per epoch and using reduced validation batches
        return [
            ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", save_last=True),
            EarlyStopping(monitor="val_loss", patience=self.stopping_patience),
        ]

    def state_dict(self, *args, **kwargs) -> dict:
        # We override state_dict so that the wrapped class can be loaded directly from this dict
        # without instantiating the LitClassifier wrapper.
        d = self.model.state_dict(*args, **kwargs)
        d["hparams"] = self.hparams
        return d

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ):
        if "hparams" in state_dict:
            self.hparams.update(state_dict.pop("hparams"))
        self.model.load_state_dict(state_dict, strict=strict, assign=assign)
