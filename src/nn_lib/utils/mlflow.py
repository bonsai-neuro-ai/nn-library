import tempfile
from pathlib import Path
from typing import Union, Optional, Any

import mlflow
import pandas as pd
import torch
from mlflow.entities import Run

from .cli import ParamsLike, NestedKey, flatten_params

RunOrURI = Union[pd.Series, Run, str, Path]


def log_flattened_params(params: ParamsLike, ignore: NestedKey = None):
    """Log the given parameters to the current MLflow run. If the parameters are a Namespace,
    they will be converted to a dictionary first. Nested parameters are flattened.
    """
    mlflow.log_params(flatten_params(params, ignore=ignore))


def search_runs_by_params(
    experiment_name: str,
    params: Optional[ParamsLike] = None,
    tags: Optional[ParamsLike] = None,
    tracking_uri: Optional[Union[str, Path]] = None,
    finished_only: bool = True,
    ignore: NestedKey = None,
) -> pd.DataFrame:
    """Query the MLflow server for runs in the specified experiment that match the given
    parameters (which will be flattened if they aren't already). Keys in `ignore` will be ignored.
    """

    def _quote_value(val: Any):
        val = str(val)
        has_single_quote = "'" in val
        has_double_quote = '"' in val
        if has_single_quote and has_double_quote:
            # Todo: figure out how to escape characters in values. MLFlow docs seem to imply it
            #  should be supported, but I can't get it to work.
            raise ValueError(
                "Parameter value containing both single and double quotes will be a problem "
                "for MLFlow filter strings"
            )
        if has_single_quote:
            return f'"{val}"'
        else:
            return f"'{val}'"

    query_parts = []
    if params is not None:
        flattened_params = flatten_params(params, ignore)
        query_parts.extend(
            [
                f"params.`{k}` = {_quote_value(v)}"
                for k, v in flattened_params.items()
                if v is not None
            ]
        )
    if tags is not None:
        flattened_tags = flatten_params(tags, ignore)
        query_parts.extend(
            [f"tags.`{k}` = {_quote_value(v)}" for k, v in flattened_tags.items() if v is not None]
        )
    if finished_only:
        query_parts.append("status = 'FINISHED'")
    query_string = " and ".join(query_parts)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.search_runs(experiment_names=[experiment_name], filter_string=query_string)


def search_single_run_by_params(
    experiment_name: str,
    params: Optional[ParamsLike] = None,
    tags: Optional[dict] = None,
    tracking_uri: Optional[Union[str, Path]] = None,
    finished_only: bool = True,
    ignore: NestedKey = None,
) -> pd.Series:
    """Query the MLflow server for runs in the specified experiment that match the given parameters.
    If exactly one run is found, return it. If no runs or multiple runs are found, raise an error.
    """
    df = search_runs_by_params(experiment_name, params, tags, tracking_uri, finished_only, ignore)
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
