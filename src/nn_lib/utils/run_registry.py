"""Parameterized MLflow runs: canonical config logging + fast deduplication.

Design
------
- A run is identified by its *spec*: a mapping (or jsonargparse.Namespace) of **identity**
  parameters -- the things that make this run this run. Execution details (device, batch_size,
  num_workers, data_root) should never be in the spec.
- Specs are canonicalized (Path -> str, Enum -> value, ...) and dumped with `yaml.safe_dump`. This
  means we're serializing arbitrary input arguments, so we cannot guarantee 100% match between
  original runs and re-runs from the dumped config. But we do the best we can.
- Deduplication is one MLflow query per process. Each run logged through `logged_run` carries a
  `run_registry.params_hash` tag; the index matches on that tag first, so correctness does not
  depend on reproducing MLflow's param serialization (which stringifies with `str()` and silently
  truncates values over 6000 chars).

Typical usage::

    specs = [dict(modelA=..., layerA=a, layerB=b, ...) for a, b in pairs]
    run_index = RunIndex.from_experiment(keys=specs[0].keys())
    to_run = [s for s in specs if s not in run_index]
    for spec in to_run:
        with logged_run(spec, index=run_index):
            ...compute...
            mlflow.log_metric("distance", d)
"""

import dataclasses
import hashlib
import re
import traceback
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

import mlflow
import yaml
from mlflow.entities import Run

try:
    from jsonargparse import Namespace
except ImportError:  # jsonargparse optional; plain dicts work fine
    Namespace = ()  # isinstance(x, ()) is always False


PARAMS_HASH_MLFLOW_TAG = "run_registry.params_hash"


#######################
# Canonical/flat form #
#######################


# Matches hex memory addresses as they appear in default object/function reprs,
# e.g. "<Foo object at 0x7f3a2c1d5e80>" or "<function f at 0x7f...>".
_MEMORY_ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")


def to_plain(params: "Namespace | Mapping[str, Any]") -> dict[str, Any]:
    """Convert a spec to a plain nested dict of yaml-safe values."""
    if isinstance(params, Namespace):
        params = params.as_dict()
    return {str(k): _plain_value(v) for k, v in params.items()}


def _plain_value(v: Any) -> Any:
    if isinstance(v, Namespace):  # nested jsonargparse Namespace
        v = v.as_dict()
    if isinstance(v, Mapping):
        return {str(k): _plain_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain_value(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted((_plain_value(x) for x in v), key=str)
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, Path):
        return str(v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        # Mirror jsonargparse's class_path/init_args serialization convention.
        return {
            "class_path": f"{type(v).__module__}.{type(v).__qualname__}",
            "init_args": {f.name: _plain_value(getattr(v, f.name)) for f in dataclasses.fields(v)},
        }
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    # Fallback for arbitrary objects: str(v) serialization is acceptable only if deterministic
    # across processes. Default object reprs (and function/lambda/partial reprs) embed memory
    # addresses, which would make hashes never match existing runs and silently break
    # deduplication -- fail loudly instead.
    s = str(v)
    if _MEMORY_ADDRESS.search(s):
        raise TypeError(
            f"Cannot canonicalize {type(v).__qualname__} value {s!r}: its string form contains a "
            f"memory address, so its hash would differ between processes and deduplication would "
            f"silently break. Build specs from serializable values instead. Recommended: pass the "
            f"pre-instantiation jsonargparse config (before parser.instantiate_classes), or give "
            f"this class a deterministic __str__ that can be used for uniqueness/deduplication "
            f"purposes."
        )
    return s


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts with dotted keys, e.g. {"a": {"b": 1}} -> {"a.b": 1}.

    This matches jsonargparse's `Namespace.items()` dotted-key convention, but is deliberately
    not replaced by it: `Namespace.items()` only flattens Namespace nesting and treats plain-dict
    values as opaque leaves, so a CLI-parsed config and an equivalent program-built dict spec
    would flatten to different keys (and hash differently). Flattening the plain-dict form gives
    both Namespace-style and dict-style configs identical results.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten(v, prefix=key + "."))
        else:
            if key in out:
                raise ValueError(f"Key collision in flattened dict: {key}")
            out[key] = v
    return out


def canonical_strings(params: "Namespace | Mapping[str, Any]") -> dict[str, str]:
    """Flat {key: str(value)} form -- exactly how MLflow stores logged params."""
    return {k: str(v) for k, v in flatten(to_plain(params)).items()}


def dump_config_yaml(params: "Namespace | Mapping[str, Any]") -> str:
    return yaml.safe_dump(to_plain(params), sort_keys=True)


###############################################
# Deduplication of runs with identical params #
###############################################


def spec_hash(strings: Mapping[str, str], keys: Iterable[str]) -> str:
    blob = "\n".join(f"{k}={strings[k]}" for k in sorted(keys))
    return hashlib.sha256(blob.encode()).hexdigest()


def hash_spec(spec: "Namespace | Mapping[str, Any]", keys: Optional[Iterable[str]] = None) -> str:
    """Hash a spec's canonical string form. If `keys` is None, all keys are used."""
    strings = canonical_strings(spec)
    if keys is None:
        keys = strings.keys()
    else:
        missing = sorted(set(keys) - strings.keys())
        if missing:
            raise KeyError(f"Spec is missing identity keys {missing}")
    return spec_hash(strings, keys)


class RunIndex:
    """In-memory index of existing runs, keyed by a hash of their (string) params.

    Build it once per process with :meth:`from_experiment` (a single MLflow query),
    then membership tests are O(1) and offline.

    :param keys: the identity of a run, given either as an example spec (Namespace or mapping --
        recommended, since the canonical flat keys like "metric.init_args.p" are derived for you) or
        as an iterable of canonical flat key strings. All candidate specs must share this key set
        (the usual grid-of-jobs situation).
    """

    def __init__(self, keys: "Namespace | Mapping[str, Any] | Iterable[str]"):
        if isinstance(keys, (Namespace, Mapping)):
            keys = canonical_strings(keys).keys()
        self.keys: tuple[str, ...] = tuple(sorted(map(str, keys)))
        self._hash_to_run_id: dict[str, str] = {}

    @classmethod
    def from_experiment(
        cls,
        keys: "Namespace | Mapping[str, Any] | Iterable[str]",
        statuses: Iterable[str] = ("FINISHED", "RUNNING"),
        experiment_names: Optional[list[str]] = None,
    ) -> "RunIndex":
        """One query to MLflow; indexes all runs whose params cover `keys`.

        By default, both FINISHED and RUNNING runs count as duplicates, so parallel workers
        launched against the same grid won't double-book (modulo a small startup race -- see
        `logged_run(index=...)`).
        """
        index = cls(keys)
        statuses = set(statuses)
        runs: list[Run] = mlflow.search_runs(
            experiment_names=experiment_names, output_format="list"
        )
        for run in runs:
            if run.info.status in statuses:
                index.add_existing_run(run)
        return index

    def add_existing_run(self, run: Run) -> None:
        # For jobs run with `logged_run`, we populate the PARAMS_HASH_MLFLOW_TAG ourselves and this
        # reliably gives us a way to track specs/params/hashes.
        tag = run.data.tags.get(PARAMS_HASH_MLFLOW_TAG)
        if tag:
            self._hash_to_run_id.setdefault(tag, run.info.run_id)

        # Fallback: maybe this is an old experiment or maybe it wasn't run with `logged_run`. So
        # we do our best to reconstruct what the hash would have been from stored params. This
        # essentially assumes that what we get from spec_hash of canonical_strings of a raw spec
        # matches what mlflow would have logged.
        params = run.data.params
        if all(k in params for k in self.keys):
            h = spec_hash(params, self.keys)
            self._hash_to_run_id.setdefault(h, run.info.run_id)

    def add(self, spec: "Namespace | Mapping[str, Any]", run_id: str = "") -> None:
        self._hash_to_run_id.setdefault(self._hash(spec), run_id)

    def lookup(self, spec: "Namespace | Mapping[str, Any]") -> Optional[str]:
        """Return the run_id of an existing matching run, or None."""
        return self._hash_to_run_id.get(self._hash(spec))

    def __contains__(self, spec: "Namespace | Mapping[str, Any]") -> bool:
        return self._hash(spec) in self._hash_to_run_id

    def __len__(self) -> int:
        return len(self._hash_to_run_id)

    def pending(self, specs: Iterable[Any]) -> Iterator[Any]:
        """Yield only the specs that don't already have a run."""
        for spec in specs:
            if spec not in self:
                yield spec

    def _hash(self, spec: "Namespace | Mapping[str, Any]") -> str:
        return hash_spec(spec, self.keys)


#####################################
# Run management with auto-indexing #
#####################################


@contextmanager
def logged_run(
    spec: "Namespace | Mapping[str, Any]",
    index: Optional[RunIndex] = None,
    **start_run_kwargs,
):
    """`with logged_run(spec) as run:` == `with mlflow.start_run() as run:` plus:

    - logs the spec via `mlflow.log_params`. (Does *not* log params outside the spec, such as
      any system config like GPU type or CPU count; the caller is responsible for logging these
      separately if they want to).
    - logs the spec as a `config.yaml` artifact
    - tags the run with a hash of the spec (`PARAMS_HASH_MLFLOW_TAG`) so future deduplication
      doesn't depend on MLflow's param serialization
    - on exception, logs the traceback as an `error.log` artifact (mlflow.start_run's own
      __exit__ then marks the run FAILED)
    - if `index` is given, registers this run in it immediately, so the same process can never
      start a duplicate even if this run later fails. The hash is computed over `index.keys` when
      an index is given (spec may not contain extra keys beyond them), else over all spec keys.
    """
    plain_spec = to_plain(spec)
    strings = canonical_strings(spec)
    keys = index.keys if index is not None else tuple(sorted(strings.keys()))
    if set(strings.keys()) - set(keys):
        raise KeyError(
            f"Spec has keys {sorted(set(strings.keys()) - set(keys))} not in the "
            "index; specs and index must agree on the identity key set."
        )
    h = spec_hash(strings, keys)
    with mlflow.start_run(**start_run_kwargs) as run:
        mlflow.log_params(flatten(plain_spec))
        mlflow.log_text(yaml.safe_dump(plain_spec, sort_keys=True), "config.yaml")
        mlflow.set_tag(PARAMS_HASH_MLFLOW_TAG, h)
        if index is not None:
            index.add(spec, run.info.run_id)
        try:
            yield run
        except BaseException:
            mlflow.log_text(traceback.format_exc(), "error.log")
            raise


__all__ = [
    "logged_run",
    "RunIndex",
]
