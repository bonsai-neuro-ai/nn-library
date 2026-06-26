import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Union, Optional, Any, Literal, Iterable, Generator

import mlflow
import pandas as pd
import torch
from jsonargparse import ArgumentParser, Namespace
from mlflow.entities import Run

RunOrURI = Union[pd.Series, Run, str, Path]


class RunExists(Exception):
    def __init__(self, run: Run, *args):
        super().__init__(*args)
        self.run = run


class RunDoesNotExist(Exception):
    pass


@contextmanager
def open_mlflow_artifact_file(
    path: Union[str, Path], mode: str = "w", run_id: Optional[str] = None
):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / path
        with open(local_file, mode) as f:
            yield f
        mlflow.log_artifact(str(local_file), artifact_path=str(path), run_id=run_id)


def log_params_and_config(params: Namespace, parser: ArgumentParser):
    """Log the given parameters (a jsonargparse Namespace) to the current MLFlow run *and* log a
    'config.yaml' file as an MLFlow artifact.
    """
    mlflow.log_params(flatten_params(params))
    mlflow.log_text(parser.dump(params, format="yaml"), "config.yaml")


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


# TODO a run can have status 'FINISHED' or 'ERROR' or 'RUNNING'. We should make it an option to
#  exclude errors but include 'RUNNING'. In other words, finished_only shouldn't just be bool. But
#  changing this behavior could break some things. Need to think about backwards-compatibility.
def _build_filter_string(
    params: Optional[Namespace] = None,
    finished_only: bool = True,
    skip_keys: Optional[Iterable[str]] = None,
) -> str:
    query_parts = []
    if params is not None:
        flattened_params = flatten_params(params, skip_keys)
        query_parts.extend(
            [
                f"params.`{k}` = {_quote_value(v)}"
                for k, v in flattened_params.items()
                if v is not None
            ]
        )
    if finished_only:
        query_parts.append("status = 'FINISHED'")
    return " and ".join(query_parts)


def search_runs_by_params(
    params: Optional[Namespace] = None,
    experiment_name: Optional[str | list[str]] = None,
    tracking_uri: Optional[Union[str, Path]] = None,
    finished_only: bool = True,
    skip_keys: Optional[Iterable[str]] = None,
    output_format: Literal["pandas", "list"] = "pandas",
) -> list[Run] | pd.DataFrame:
    """Query the MLflow server for runs in the specified experiment that match the given
    parameters (which will be flattened if they aren't already). Keys in `skip_keys` will be ignored.
    """
    query_string = _build_filter_string(params, finished_only, skip_keys)
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
    params: Optional[Namespace] = None,
    experiment_name: Optional[str | list[str]] = None,
    tracking_uri: Optional[Union[str, Path]] = None,
    finished_only: bool = True,
    skip_keys: Optional[Iterable[str]] = None,
) -> Run:
    """Query the MLflow server for runs in the specified experiment that match the given parameters.
    If exactly one run is found, return it. If no runs or multiple runs are found, raise an error.
    """
    runs: list[Run] = search_runs_by_params(
        params, experiment_name, tracking_uri, finished_only, skip_keys, output_format="list"
    )
    if len(runs) == 0:
        raise RunDoesNotExist("No runs found with the specified parameters")
    elif len(runs) > 1:
        raise RunExists(runs[0], "Multiple runs found with the specified parameters")
    return runs[0]


def run_has_params(run: Run, params: Namespace, skip_keys: Optional[Iterable[str]] = None) -> bool:
    """Check if a Run has parameters. This provides an alternative to 'search_runs_by_params' where
    runs can be pre-fetched from the MLflow server and compared in memory to params.
    """

    run_params = run.data.params
    for k, v in _iter_params_skip(params, skip_keys):
        # Internally, mlflow wraps all params in a str() call, so we need to check equality vs
        # the stringified param (which could be any data type). See mlflow.log_params for reference.
        if k not in run_params or run_params[k] != str(v):
            return False
    return True


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


def _match_nested_key_prefix(key: str, prefix: str) -> bool:
    """Check if a maybe-dot-separated prefix like 'foo' matches a maybe-dot-separated key like
    'foo.bar.baz'. This is used to skip nested keys when flattening a Namespace. Note that dot
    separators are important and 'f' is not considered a prefix of 'foo' or 'foo.bar', but 'f' *is*
    a prefix of 'f.x.y.z'.
    """
    key_parts = key.split(".")
    prefix_parts = prefix.split(".")
    if len(prefix_parts) > len(key_parts):
        return False
    for kp, pp in zip(key_parts, prefix_parts):
        if kp != pp:
            return False
    return True


def _iter_params_skip(
    params: Namespace, skip_keys: Optional[Iterable[str]] = None
) -> Generator[tuple[str, Any], None, None]:
    """Like Namespace.items(), iterate leaf (key, value) options of a possibly-nested Namespace.
    For example, if `params` is like `Namespace(a=1, b=Namespace(c=2, d=3))`, this will yield
    `('a', 1)`, `('b.c', 2)`, and `('b.d', 3)`.

    'skip_keys' can specify fields to skip. For instance if `skip_keys = ['a']` then we just get
    `('b.c', 2)`, and `('b.d', 3)`. If `skip_keys = ['b']` we get just `('a', 1)`. Skipped keys can
    also be nested, so if `skip_keys = ['a', 'b.c']` we'll get back just `('b.d', 3)`. If
    `skip_keys = None` (the default), nothing is skipped and all elements from `params.items()` are
    yielded.
    """
    skip_keys = set() if skip_keys is None else set(skip_keys)
    for nested_key, value in params.items():
        if any(_match_nested_key_prefix(nested_key, skip) for skip in skip_keys):
            continue
        yield nested_key, value


def flatten_params(params: Namespace, skip_keys: Optional[Iterable[str]] = None) -> dict:
    """Flatten the given parameters, like Namespace.as_flat, but allow some keys to be skipped and
    returning as a dict."""
    return dict(_iter_params_skip(params, skip_keys))


def auto_cli_mlflow_job(
    run_fn,
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment: str = "debug",
    deduplicate: bool = True,
    ignore_keys_dedup: Optional[Iterable[str]] = None,
    async_logging: bool = True,
    log_system_metrics: bool = True,
):
    """Analogous to the jsonargparse auto_cli function, this is an automatic CLI tool which also
    handles the typical mlflow boilerplate we use.

    Usage: in some `my_script.py` file, define a main function which runs things:

        def main(foo: str, bar: int = 4):
            # inside here, do computation and log stuff
            mlflow.log_text("foo")
            mlflow.log_metric("bar", bar)

        if __name__ == "__main__":
            auto_cli_mlflow_job(
                main,
                tracking_uri="sqlite:////projects/my-project-name/mlflow.db",
                experiment="my_experiment"
            )

    And in a shell script:

        python my_script.py --foo hello --bar 5

    Or on a Slurm cluster, do a '--dry_run' to determine if the job should be run, and if so, queue
    it with srun:

        ARGS = "--foo hello --bar 5"
        python my_script.py $ARGS --dry_run || srun python my_script.py $ARGS

    :param run_fn: the function to run. CLI arguments are inferred from its default arguments and
        type annotations. See jsonargparse docs for details, namely how
        `add_function_arguments(run_fn)` behaves.
    :param tracking_uri: the mlflow tracking URI. Should be unique per project.
    :param experiment: the mlflow experiment in which this fn call will appear as a single run.
    :param deduplicate: if True, check if a run with the same parameters already exists in the
        experiment and exit early.
    :param ignore_keys_dedup: config keys to ignore for the purpose of deduplication. Some
        examples could be if you have a 'seed' argument or a 'device' argument where two runs with
        different values should actually be considered copies of each other. See `flatten_params`
        for details on how nested keys are handled.
    :param async_logging: turn on async logging in mlflow (default True)
    :param log_system_metrics: turn on system metrics logging in mlflow (default True)
    """
    # Parser for the function's own arguments
    parser = ArgumentParser()
    parser.add_function_arguments(run_fn)
    parser.add_argument("--config", action="config")
    parser.add_argument("--mlflow_tracking_uri", default=tracking_uri)
    parser.add_argument("--mlflow_experiment", default=experiment)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # All args that are not passed to the run_fn must be popped. A combined config will be
    # written to a file. Popping the 'config' field here means that we're not storing the path to
    # whatever partial config file may have been used as input to the script.
    args.pop("config")
    dry_run = args.pop("dry_run")
    mlflow.set_tracking_uri(args.pop("mlflow_tracking_uri"))
    mlflow.set_experiment(args.pop("mlflow_experiment"))

    if async_logging:
        mlflow.config.enable_async_logging()

    if log_system_metrics:
        mlflow.config.enable_system_metrics_logging()

    fn_args_instantiated = parser.instantiate(args)
    try:
        with fancy_start_run(args, deduplicate, ignore_keys_dedup):
            log_params_and_config(args, parser)
            run_fn(**fn_args_instantiated.as_dict())
    except RunExists:
        if dry_run:
            # exit with status 1 so that shell scripts can use the
            #     python script.py $ARGS --dry_run || srun python script.py $ARGS
            # pattern
            exit(1)


class fancy_start_run(object):
    def __init__(
        self,
        args: Namespace,
        deduplicate: bool,
        ignore_keys_dedup: Optional[Iterable[str]] = None,
        **start_run_kwargs,
    ):
        self._args = args
        self._deduplicate = deduplicate
        self._ignore_keys_dedup = ignore_keys_dedup
        self._start_run_kwargs = start_run_kwargs

    def __enter__(self):
        if self._deduplicate:
            existing: list[Run] = search_runs_by_params(
                params=self._args,
                finished_only=False,  # TODO skip RUNNING too
                skip_keys=self._ignore_keys_dedup,
                output_format="list",
            )
            if len(existing) > 0:
                raise RunExists(existing[0])

        # These next two lines are 'as if' we've done `with mlflow.start_run() as self._the_run:`
        # but we will handle the exiting of the `with` in our own __exit__ function.
        self._the_run = mlflow.start_run(**self._start_run_kwargs)
        self._the_run.__enter__()

        # --- starting here it's as-if we're inside the 'with mlflow.start_run' block ---
        return self._the_run

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
    "RunDoesNotExist",
    "RunExists",
    "auto_cli_mlflow_job",
    "flatten_params",
    "load_artifact",
    "log_params_and_config",
    "open_mlflow_artifact_file",
    "save_as_artifact",
    "search_runs_by_params",
    "search_single_run_by_params",
]
