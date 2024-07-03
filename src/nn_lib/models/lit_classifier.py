from torch import nn
import lightning as lit
from nn_lib.models.graph_module import GraphModule, ModelType
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy


class LitClassifier(lit.LightningModule):
    """A simple classification class. Provided with a GraphModule architecture, this class wraps it
    in a LightningModule to train it using some standard training logic.
    """
    def __init__(self, architecture: ModelType, num_classes: int, last_layer_name: str):
        super().__init__()
        self.model = GraphModule(architecture)
        self.loss = nn.CrossEntropyLoss()
        self.metrics = {"acc": Accuracy("multiclass", num_classes=num_classes)}
        self._last_layer_name = last_layer_name

    def forward(self, x):
        return self.model(x)

    def to(self, device):
        out = super().to(device)
        for metric in self.metrics.values():
            metric.to(device)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)[self._last_layer_name]
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        for name, metric in self.metrics.items():
            self.log(f"train_{name}", metric(y_hat, y))
        return loss

    def on_train_epoch_start(self):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)[self._last_layer_name]
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
