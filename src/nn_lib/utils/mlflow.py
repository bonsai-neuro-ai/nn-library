import importlib
import tempfile
from pathlib import Path
from typing import Union, Optional, Iterable, assert_never, Generator

import jsonargparse
import mlflow
import pandas as pd
import torch
from mlflow.entities import Run
from torch import nn

from nn_lib.utils import iter_flatten_dict

RunOrURI = Union[pd.Series, Run, str, Path]


def restore_params_from_mlflow_run(mlflow_run: pd.Series):
    """MLflow runs are loaded as pandas Series where the column names starting with 'params.' tell
    us the values originally stored in MLFlowLogger.log_hyperparams(). This function recovers the
    original dictionary of hyperparameters from the stored_params Series.

    For example, if the run was created with

    ```
    mlflow.log_params({"a": 1, "b": {"c": 2, "d": 3}})
    ```

    then the stored_params Series will have columns `params.a` and `params.b.c` and `params.b.d` with
    ```

    Then the stored_params Series will have columns `params.a`, `params.b/c` and `params.b/d` with
    values 1, 2, and 3 respectively. Given that row, this function will return the dictionary.
    """
    params = {}
    for col, value in mlflow_run.items():
        if isinstance(col, str) and col.startswith("params."):
            keys = col[len("params.") :].split("/")
            d = params
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
    return params


def search_runs_by_params(
    experiment_name: str,
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
    tracking_uri: Optional[Union[str, Path]] = None,
    finished_only: bool = True,
    skip_fields: Optional[dict] = None,
) -> pd.DataFrame:
    """Query the MLflow server for runs in the specified experiment that match the given
    parameters. Any keys of the `meta_fields` dictionary will be excluded from the search."""
    query_parts = []
    if params is not None:
        flattened_params = dict(iter_flatten_dict(params, join_op="/".join, skip_keys=skip_fields))
        query_parts.extend(
            [f"params.`{k}` = '{v}'" for k, v in flattened_params.items() if v is not None]
        )
    if tags is not None:
        flattened_tags = dict(iter_flatten_dict(tags, join_op="/".join, skip_keys=skip_fields))
        query_parts.extend(
            [f"tags.`{k}` = '{v}'" for k, v in flattened_tags.items() if v is not None]
        )
    if finished_only:
        query_parts.append("status = 'FINISHED'")
    query_string = " and ".join(query_parts)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.search_runs(experiment_names=[experiment_name], filter_string=query_string)


def search_single_run_by_params(
    experiment_name: str,
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
    tracking_uri: Optional[Union[str, Path]] = None,
    finished_only: bool = True,
    skip_fields: Optional[dict] = None,
) -> pd.Series:
    """Query the MLflow server for runs in the specified experiment that match the given parameters.
    If exactly one run is found, return it. If no runs or multiple runs are found, raise an error.
    """
    df = search_runs_by_params(
        experiment_name, params, tags, tracking_uri, finished_only, skip_fields
    )
    if len(df) == 0:
        raise ValueError("No runs found with the specified parameters")
    elif len(df) > 1:
        raise ValueError("Multiple runs found with the specified parameters")
    return df.iloc[0]


def instantiate(model_class: str, init_args: dict) -> object:
    """Take a string representation of a class and a dictionary of arguments and instantiate the
    class with those arguments.
    """
    model_package, model_class = model_class.rsplit(".", 1)
    cls = getattr(importlib.import_module(model_package), model_class)

    parser = jsonargparse.ArgumentParser(exit_on_error=False)
    parser.add_class_arguments(cls, nested_key="obj", instantiate=True)
    parsed = parser.parse_object({"obj": init_args})
    return parser.instantiate_classes(parsed).obj


def load_checkpoint_from_mlflow_run(
    run_or_uri: RunOrURI,
    alias: str = "best",
    map_location: Optional[str] = None,
):
    # mlflow checkpoint artifacts are stored like checkpoints/epoch-4-step=10/epoch-4-step=10.ckpt
    # with checkpoints/epoch-4-step=10/aliases.txt containing aliases like 'best' or 'last'.
    for file in _iter_artifacts(run_or_uri):
        if file.name == "aliases.txt":
            with open(file, "r") as f:
                if f"'{alias}'" in f.read():
                    the_checkpoint = file.parent / f"{file.parent.name}.ckpt"
                    break
    else:
        raise FileNotFoundError(f"Could not find checkpoint with alias '{alias}'")
    return torch.load(the_checkpoint, map_location=map_location)


def restore_model_from_mlflow_run(
    run: pd.Series,
    load_checkpoint: bool = True,
    device: Optional[str] = None,
    alias: str = "best",
    drop_keys: Optional[Iterable[str]] = None,
):
    params_dict = restore_params_from_mlflow_run(run)
    model_class = params_dict["model"]["class_path"]
    init_args = params_dict["model"]["init_args"]

    model: nn.Module = instantiate(model_class, init_args)  # type: ignore

    if load_checkpoint:
        data = load_checkpoint_from_mlflow_run(run, alias=alias, map_location=device)
        if drop_keys:
            data["state_dict"] = {k: v for k, v in data["state_dict"].items() if k not in drop_keys}
        model.load_state_dict(data["state_dict"])

    return model


def restore_data_from_mlflow_run(run: pd.Series):
    params_dict = restore_params_from_mlflow_run(run)
    model_class = params_dict["data"]["class_path"]
    init_args = params_dict["data"]["init_args"]

    return instantiate(model_class, init_args)


def save_as_artifact(obj: object, path: Path, run_id: Optional[str] = None):
    """Save the given object to the given path as an MLflow artifact in the given run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / path.name
        remote_path = str(path.parent) if path.parent != Path() else None
        torch.save(obj, local_file)
        mlflow.log_artifact(str(local_file), artifact_path=remote_path, run_id=run_id)


def _to_mlflow_uri(run_or_uri: RunOrURI) -> str:
    """Canonicalize the given run or URI to a string representation of the URI."""
    match run_or_uri:
        case str():
            return run_or_uri
        case Path():
            return str(run_or_uri)
        case pd.Series():
            return run_or_uri.artifact_uri
        case Run():
            return run_or_uri.info.artifact_uri
    assert_never(run_or_uri)


def _iter_artifacts(run_or_uri: RunOrURI) -> Generator[Path, None, None]:
    """Iterate over all files in the given MLflow run's artifact URI."""
    for file in Path(_to_mlflow_uri(run_or_uri)).iterdir():
        if file.is_file():
            yield file
        elif file.is_dir():
            yield from _iter_artifacts(file)



__all__ = [
    "instantiate",
    "load_checkpoint_from_mlflow_run",
    "restore_data_from_mlflow_run",
    "restore_model_from_mlflow_run",
    "restore_params_from_mlflow_run",
    "save_as_artifact",
    "search_runs_by_params",
    "search_single_run_by_params",
]
