import tempfile
from enum import Enum, auto
from pathlib import Path
import mlflow
import torch


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

    def __str__(self):
        return self.name
