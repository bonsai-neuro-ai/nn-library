import torch
import lightning as lit
import pandas as pd
import mlflow
import importlib
import inspect
from pathlib import Path
from typing import Optional, Callable, Generator, Tuple, Any, TypeVar, Iterable, Union


T = TypeVar("T")
K = TypeVar("K")
J = TypeVar("J")


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
    params: dict,
    tracking_uri: Optional[Union[str, Path]] = None,
    finished_only: bool = True,
    meta_fields: Optional[dict] = None,
) -> pd.DataFrame:
    """Query the MLflow server for runs in the specified experiment that match the given parameters."""
    flattened_params = dict(iter_flatten_dict(params, join_op="/".join, skip_keys=meta_fields))
    query_parts = [f"params.`{k}` = '{v}'" for k, v in flattened_params.items() if v is not None]
    if finished_only:
        query_parts.append("status = 'FINISHED'")
    query_string = " and ".join(query_parts)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.search_runs(experiment_names=[experiment_name], filter_string=query_string)


def search_single_run_by_params(
    experiment_name: str,
    params: dict,
    tracking_uri: Optional[Union[str, Path]] = None,
    finished_only: bool = True,
) -> pd.Series:
    """Query the MLflow server for runs in the specified experiment that match the given parameters.
    If exactly one run is found, return it. If no runs or multiple runs are found, raise an error.
    """
    df = search_runs_by_params(experiment_name, params, tracking_uri, finished_only)
    if len(df) == 0:
        raise ValueError("No runs found with the specified parameters")
    elif len(df) > 1:
        raise ValueError("Multiple runs found with the specified parameters")
    return df.iloc[0]


def instantiate(model_class: str, init_args: dict) -> object:
    """Take a string representation of a class and a dictionary of arguments and instantiate the
    class with those arguments.
    """
    # TODO – this is duplicating work already done by jsonargparse. See if we can redirect to
    #  something we can import from there.
    model_package, model_class = model_class.rsplit(".", 1)
    cls = getattr(importlib.import_module(model_package), model_class)
    # inspect the signature of the class constructor and check types
    sig = inspect.signature(cls.__init__)
    for param_name, param_value in init_args.items():
        param_type = sig.parameters[param_name].annotation
        if param_type == inspect.Parameter.empty:
            raise ValueError(f"Parameter {param_name} is missing annotation in {cls}")
        if not isinstance(param_value, param_type):
            try:
                # Attempt to cast the parameter to the correct type
                init_args[param_name] = param_type(param_value)
            except Exception as e:
                raise ValueError(
                    f"Parameter {param_name} should be of type {param_type} but got {param_value}"
                ) from e
    return cls(**init_args)


def _iter_files(path: Union[Path, str]):
    # TODO - this is using path traversal when we should probably be using the MLflow API (I tried,
    #  but mlflow.artifacts.list_artifacts() gave inconsistently formatted paths.)
    for file in Path(path).iterdir():
        if file.is_file():
            yield file
        elif file.is_dir():
            yield from _iter_files(file)


def restore_model_from_mlflow_run(
    run: pd.Series, load_checkpoint: bool = True, device: Optional[str] = None
):
    # TODO - configurable which checkpoint to load and/or load "best" checkpoint
    params_dict = restore_params_from_mlflow_run(run)
    model_class = params_dict["model"]["class_path"]
    init_args = params_dict["model"]["init_args"]

    model: lit.LightningModule = instantiate(model_class, init_args)  # type: ignore

    if load_checkpoint:
        checkpoints = [
            file for file in _iter_files(run["artifact_uri"]) if file.name.endswith(".ckpt")
        ]
        data = torch.load(checkpoints[-1], map_location=device)
        model.load_state_dict(data["state_dict"])

    return model


def restore_data_from_mlflow_run(run: pd.Series):
    params_dict = restore_params_from_mlflow_run(run)
    model_class = params_dict["data"]["class_path"]
    init_args = params_dict["data"]["init_args"]

    return instantiate(model_class, init_args)


def iter_flatten_dict(
    d: dict[K, Any],
    join_op: Callable[[tuple[K, ...]], J],
    prefix: Tuple[K, ...] = tuple(),
    skip_keys: Optional[Union[Iterable[K], dict[K, Any]]] = None,
) -> Generator[Tuple[str, Any], None, None]:
    """Iterate a nested dict in order, yielding (k1k2k3, v) from a dict like {k1: {k2: {k3: v}}}.
    Uses the given join_op to join keys together. In this example, join_op(k1, k2, k3) should
    return k1k2k3
    """
    skip_keys = skip_keys or {}
    for k, v in d.items():
        if k in skip_keys and not isinstance(skip_keys, dict):
            continue
        new_prefix = prefix + (k,)
        if type(v) is dict:
            yield from iter_flatten_dict(
                v, join_op, new_prefix, skip_keys[k] if k in skip_keys else None
            )
        else:
            joined_key = join_op(new_prefix)
            yield joined_key, v
