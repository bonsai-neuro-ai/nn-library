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


class mlflow_start_singleton_run(object):
    """Fancier replacement for mlflow.start_run. Instead of all this:

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.enable_async_logging()

        if run_exists(params, ignore):
            return

        with mlflow.start_run(**kw) as run:
            mlflow.log_params(flatten_params(params, ignore))
            try:
                # do stuff
            except Exception as e:
                log_error(e)

    ...instead just do this:

        try:
            with mlflow_start_singleton_run(params, ignore, experiment_name, tracking_uri, **kw) as run:
                # do stuff
        except RunExists:
            print("skipping!")

    ...and this will provide the following features:

    1. set experiment name and tracking uri if provided (warning: global!)
    2. check if run exists with these parameters and if so just return it without doing stuff
    3. configure logging to dump stuff to artifacts (out.log and err.log)
    4. start the run
    5. both log params and save as a config.yaml artifact

    Additional kwargs are passed through to mlflow.start_run(). See `search_runs_by_params` for
    details on how 'params' and 'ignore' interact.
    """

    _the_run: ActiveRun

    def __init__(
        self,
        params: ParamsLike,
        ignore: NestedKey = None,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        **kwargs,
    ):
        self._params = params
        self._ignore = ignore
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._start_run_kwargs = kwargs

    def __enter__(self):
        if self._tracking_uri is not None:
            mlflow.set_tracking_uri(self._tracking_uri)

        if self._experiment_name is not None:
            mlflow.set_experiment(self._experiment_name)

        mlflow.config.enable_async_logging()

        matching_runs: list[Run] = search_runs_by_params(
            params=self._params, finished_only=True, ignore=self._ignore, output_format="list"
        )

        if matching_runs:
            raise RunExists(matching_runs[0], "Run already exists with the specified parameters")
        else:
            self._the_run = mlflow.start_run(**self._start_run_kwargs)
            self._the_run.__enter__()
            mlflow.log_params(flatten_params(self._params, ignore=self._ignore))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If there was an error, log the error as an artifact
        if self._the_run is not None:
            if exc_val is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    local_file = Path(tmpdir) / "error.log"
                    with open(local_file, "w") as f:
                        traceback.print_exception(exc_type, exc_val, exc_tb, file=f)
                    mlflow.log_artifact(local_file, run_id=self._the_run.info.run_id)
            self._the_run.__exit__(exc_type, exc_val, exc_tb)


__all__ = [
    "load_artifact",
    "log_flattened_params",
    "mlflow_start_singleton_run",
    "save_as_artifact",
    "search_runs_by_params",
    "search_single_run_by_params",
    "RunExists",
    "RunDoesNotExist",
]
