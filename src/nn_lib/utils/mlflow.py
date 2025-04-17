import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Union, Optional

import mlflow
import pandas as pd
import torch
from jsonargparse import Namespace as JSONNamespace, strip_meta
from mlflow.entities import Run

from nn_lib.utils import iter_flatten_dict

RunOrURI = Union[pd.Series, Run, str, Path]


def log_flattened_params(params: dict | Namespace | JSONNamespace):
    """Log the given parameters to the current MLflow run. If the parameters are a Namespace,
    they will be converted to a dictionary first. Nested parameters are flattened.
    """
    if isinstance(params, JSONNamespace):
        params = strip_meta(params).to_dict()
    elif isinstance(params, Namespace):
        params = vars(params)

    flattened_params = dict(iter_flatten_dict(params, join_op="/".join))
    mlflow.log_params(flattened_params)


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


def save_as_artifact(obj: object, path: str | Path, run_id: Optional[str] = None):
    """Save the given object to the given path as an MLflow artifact in the given run."""
    if isinstance(path, str):
        path = Path(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / path.name
        remote_path = str(path.parent) if path.parent != Path() else None
        torch.save(obj, local_file)
        mlflow.log_artifact(str(local_file), artifact_path=remote_path, run_id=run_id)


def load_artifact(path: str | Path, run_id: Optional[str] = None) -> object:
    """Load the given artifact from the specified MLflow run. Path is relative to the artifact URI,
    just like save_as_artifact()
    """
    if isinstance(path, Path):
        path = str(path)
    if run_id is None:
        run_id = mlflow.active_run().info.run_id
    # Note: despite the name, "downloading" artifacts involves no copying of files if we leave the
    # local path unspecified and the artifacts are stored on this file system.
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
    return torch.load(local_path)


__all__ = [
    "load_artifact",
    "log_flattened_params",
    "save_as_artifact",
    "search_runs_by_params",
    "search_single_run_by_params",
]
