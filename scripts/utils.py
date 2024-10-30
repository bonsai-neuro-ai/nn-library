import tempfile
from enum import Enum, auto
from pathlib import Path
import lightning as lit
import mlflow
import torch
from lightning.fabric.utilities import rank_zero_only
from lightning.pytorch.tuner import Tuner
from nn_lib.datasets import TorchvisionDataModuleBase
from nn_lib.models import LitClassifier


def save_as_artifact(obj: object, path: Path, run_id: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / path.name
        remote_path = str(path.parent) if path.parent != Path() else None
        torch.save(obj, local_file)
        mlflow.log_artifact(str(local_file), artifact_path=remote_path, run_id=run_id)


class JobStatus(Enum):
    RUNNING = auto()
    SUCCESS = auto()
    ERROR = auto()
    DOES_NOT_EXIST = auto()

    def __str__(self):
        return self.name


@rank_zero_only
def tune_before_training(
    tr: lit.Trainer, model: LitClassifier, dm: TorchvisionDataModuleBase
):
    """Run a tuning loop before training the model."""
    if tr.num_devices > 1:
        raise ValueError("Tuning is not supported with multi-GPU training.")

    tuner = Tuner(tr)
    tuner.lr_find(model, datamodule=dm)

    print("Tuning complete!")
    print(f"Best learning rate @ batch_size={dm.bs}:", model.lr)
