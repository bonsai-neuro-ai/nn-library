import tempfile
import traceback
from pathlib import Path
from typing import Union, Optional, Any, Literal

import mlflow
import pandas as pd
import torch
from mlflow import ActiveRun
from mlflow.entities import Run

from .cli import ParamsLike, NestedKey, flatten_params

RunOrURI = Union[pd.Series, Run, str, Path]


class RunExists(Exception):
    def __init__(self, run: Run, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run = run


class RunDoesNotExist(Exception):
    pass


def log_flattened_params(params: ParamsLike, ignore: NestedKey = None):
    """Log the given parameters to the current MLflow run. If the parameters are a Namespace,
    they will be converted to a dictionary first. Nested parameters are flattened.
    """
    mlflow.log_params(flatten_params(params, ignore=ignore))


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


def build_filter_string(
    params: Optional[ParamsLike] = None,
    tags: Optional[ParamsLike] = None,
    finished_only: bool = True,
    ignore: NestedKey = None,
) -> str:
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
    return " and ".join(query_parts)


def search_runs_by_params(
    experiment_name: Optional[str | list[str]] = None,
    params: Optional[ParamsLike] = None,
    tags: Optional[ParamsLike] = None,
    tracking_uri: Optional[Union[str, Path]] = None,
    finished_only: bool = True,
    ignore: NestedKey = None,
    output_format: Literal["pandas", "list"] = "pandas",
) -> list[Run] | pd.DataFrame:
    """Query the MLflow server for runs in the specified experiment that match the given
    parameters (which will be flattened if they aren't already). Keys in `ignore` will be ignored.
    """
    query_string = build_filter_string(params, tags, finished_only, ignore)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if experiment_name is not None:
        if isinstance(experiment_name, str):
            experiment_name = [experiment_name]
        elif isinstance(experiment_name, list):
            experiment_name = experiment_name
        else:
            raise ValueError("`experiment_name` must be a string or a list of strings")

    return mlflow.search_runs(
        experiment_names=experiment_name, filter_string=query_string, output_format=output_format
    )


def search_single_run_by_params(
    experiment_name: str,
    params: Optional[ParamsLike] = None,
    tags: Optional[dict] = None,
    tracking_uri: Optional[Union[str, Path]] = None,
    finished_only: bool = True,
    ignore: NestedKey = None,
) -> Run:
    """Query the MLflow server for runs in the specified experiment that match the given parameters.
    If exactly one run is found, return it. If no runs or multiple runs are found, raise an error.
    """
    runs = search_runs_by_params(
        experiment_name, params, tags, tracking_uri, finished_only, ignore, output_format="list"
    )
    if len(runs) == 0:
        raise RunDoesNotExist("No runs found with the specified parameters")
    elif len(runs) > 1:
        raise RunExists(runs[0], "Multiple runs found with the specified parameters")
    return runs[0]


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
    "RunExists",
    "RunDoesNotExist",
]
