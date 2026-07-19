"""Parameterized MLflow runs: canonical config logging + fast deduplication.

Design
------
- A run is identified by its *spec*: a mapping (or jsonargparse.Namespace) of **identity**
  parameters -- the things that make this run unique. Execution details (device, batch_size,
  num_workers, data_root) should never be in the spec. Terminology note: 'params' are all input
  parameters; a 'spec' is the subset of those parameters that specify unique behavior.
- Specs are canonicalized (Path -> str, Enum -> name, ...) and dumped with `yaml.safe_dump`. This
  means we're serializing arbitrary input arguments, so we cannot guarantee 100% match between
  original runs and re-runs from the dumped config. But we do the best we can.
- jsonargparse subclass-typed arguments are supported in their *pre-instantiation*
  `{class_path, init_args}` form. Live objects -- anything after `parser.instantiate_classes`,
  live-instance argument defaults, and instantiated dataclasses -- are rejected loudly, since
  canonicalizing them would silently break deduplication (see `to_plain`).
- Deduplication is one MLflow query per process. Each run logged through `logged_run` carries a
  `run_registry.params_hash` tag; the index matches on that tag first, so correctness does not
  depend on reproducing MLflow's param serialization (which stringifies with `str()` and silently
  truncates values over 6000 chars).

Typical usage::

    # Identity params only. From a CLI: spec = select_spec(parser.parse_args()).
    specs = [dict(modelA=..., layerA=a, layerB=b, ...) for a, b in pairs]
    run_index = RunIndex.from_experiment(keys=specs[0])
    for spec in run_index.pending(specs):
        with logged_run(spec, index=run_index):
            ...compute...
            mlflow.log_metric("distance", d)

Known limitations
-----------------
- All specs sharing a RunIndex must have the same flat key set. A grid over jsonargparse
  subclasses whose __init__ signatures differ (different `init_args` keys) violates this and
  fails loudly with KeyError; per-spec key sets would need a different index design. Recommended
  workaround for now is to place different init signatures in different experiments/different
  indexes.
- Empty-dict values flatten away (`{"tags": {}}` canonicalizes like a spec without "tags").
"""

import dataclasses
import hashlib
import re
import traceback
from contextlib import contextmanager
from copy import deepcopy
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

# Shared guidance appended to canonicalization errors. Live objects usually get into a spec in
# one of two ways, and the fix differs; name both so the error is actionable.
_INSTANTIATED_OBJECT_HINT = (
    "Specs must be built from pre-instantiation values; live objects usually get into a spec "
    "in one of two ways. (1) The config was passed through parser.instantiate_classes(): pass "
    "the config from parser.parse_args() instead, which keeps subclass-typed arguments in "
    "their serializable {class_path, init_args} form. (2) An argument default was declared "
    "as a live instance: declare it with jsonargparse.lazy_instance(Cls, ...) or as a "
    "{'class_path': ..., 'init_args': ...} dict so it stays serializable in parsed configs. "
    "Alternatively, if this value is an execution detail rather than part of the run's "
    "identity, drop it from the spec (see select_spec); or, for a plain class, give it a "
    "deterministic __str__ usable for uniqueness/deduplication."
)


def to_plain(params: "Namespace | Mapping[str, Any]") -> dict[str, Any]:
    """Convert a spec to a plain nested dict of yaml-safe values.

    Raises TypeError (naming the offending key) for values that cannot be canonicalized
    deterministically: objects whose str() embeds a memory address, and dataclass instances
    (whose fields alone cannot carry class identity).
    """
    if isinstance(params, Namespace):
        params = params.as_dict()
    return {str(k): _plain_value(v, _path=str(k)) for k, v in params.items()}


def _sub_path(path: str, key: Any) -> str:
    return f"{path}.{key}" if path else str(key)


def _plain_value(v: Any, _path: str = "") -> Any:
    if isinstance(v, Namespace):  # nested jsonargparse Namespace
        v = v.as_dict()
    if isinstance(v, Mapping):
        return {str(k): _plain_value(x, _path=_sub_path(_path, k)) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain_value(x, _path=f"{_path}[{i}]") for i, x in enumerate(v)]
    if isinstance(v, (set, frozenset)):
        return sorted((_plain_value(x, _path=_path) for x in v), key=str)
    if isinstance(v, Enum):
        # jsonargparse parses enums by their name, not their value.
        return v.name
    if isinstance(v, Path):
        return str(v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        # Serializing fields alone would drop class identity, so two dataclass types with
        # identical fields would silently deduplicate against each other -- fail loudly.
        raise TypeError(
            f"Cannot canonicalize dataclass instance {type(v).__qualname__} at spec key "
            f"{_path!r}: serializing its fields would drop class identity, so two dataclass "
            f"types with identical fields would silently deduplicate against each other. If "
            f"the intended class is unambiguous (a fixed-class argument), use a plain dict of "
            f"its fields instead. " + _INSTANTIATED_OBJECT_HINT
        )
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    # Fallback for arbitrary objects: str(v) serialization is acceptable only if deterministic
    # across processes. Default object reprs (and function/lambda/partial reprs) embed memory
    # addresses, which would make hashes never match existing runs and silently break
    # deduplication -- fail loudly instead.
    s = str(v)
    if _MEMORY_ADDRESS.search(s):
        raise TypeError(
            f"Cannot canonicalize {type(v).__qualname__} value {s!r} at spec key {_path!r}: "
            f"its string form contains a memory address, so its hash would differ between "
            f"processes and deduplication would silently break. " + _INSTANTIATED_OBJECT_HINT
        )
    return s


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts with dotted keys, e.g. {"a": {"b": 1}} -> {"a.b": 1}.

    This matches jsonargparse's `Namespace.items()` dotted-key convention, but is deliberately
    not replaced by it: `Namespace.items()` only flattens Namespace nesting and treats plain-dict
    values as opaque leaves, so a CLI-parsed config and an equivalent program-built dict spec
    would flatten to different keys (and hash differently). Flattening the plain-dict form gives
    both Namespace-style and dict-style configs identical results.

    Raises ValueError on dotted-key collisions like {"a.b": 1, "a": {"b": 2}}, which would
    otherwise silently merge two distinct specs into the same canonical form. Note that empty
    dict values flatten away (contribute no keys).
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        new = flatten(v, prefix=key + ".") if isinstance(v, Mapping) else {key: v}
        for new_key, new_value in new.items():
            if new_key in out:
                raise ValueError(f"Key collision in flattened dict: {new_key}")
            out[new_key] = new_value
    return out


def canonical_strings(params: "Namespace | Mapping[str, Any]") -> dict[str, str]:
    """Flat {key: str(value)} form -- exactly how MLflow stores logged params."""
    return {k: str(v) for k, v in flatten(to_plain(params)).items()}


def dump_config_yaml(params: "Namespace | Mapping[str, Any]") -> str:
    return yaml.safe_dump(to_plain(params), sort_keys=True)


def select_spec(
    params: "Namespace | Mapping[str, Any]", drop: Optional[Iterable[str]] = ("config",)
) -> "Namespace | Mapping[str, Any]":
    """Return a copy of `params` with the given top-level keys removed.

    Intended for carving a run's identity *spec* out of full CLI params: dropping the
    `--config` bookkeeping key (the default) or execution details like device/num_workers.

    :param drop: top-level keys to remove; keys not present are ignored. Defaults to
        ("config",). Pass None or an empty iterable to keep everything. Dotted/nested keys
        are not supported.
    """
    if drop is None:
        drop = []
    else:
        drop = set(drop)

    # Don't modify in-place
    params = deepcopy(params)

    if len(drop) == 0:
        return params

    for drop_key in drop:
        if "." in drop_key:
            raise NotImplementedError("Dropping nested keys is not currently supported.")
        if drop_key not in params:
            # If asking to drop something that isn't there, it's a no-op
            continue
        params.pop(drop_key)

    return params


###############################################
# Deduplication of runs with identical params #
###############################################


def _escape(s: str) -> str:
    # Without escaping, a value containing "\nother_key=..." could make two distinct specs
    # build byte-identical hash blobs (a silent dedup collision). Escaping keeps the blob
    # unambiguous while leaving hashes unchanged for specs free of backslashes/newlines.
    return s.replace("\\", "\\\\").replace("\n", "\\n")


def spec_hash(strings: Mapping[str, str], keys: Iterable[str]) -> str:
    """Deterministic hash over sorted "{key}={value}" lines for the selected keys.

    Low-level: `strings` must already be flat canonical strings (see `canonical_strings`);
    most callers want `hash_spec`. Hashes are stable across versions except for specs whose
    keys/values contain a backslash or newline (see `_escape`).
    """
    blob = "\n".join(f"{_escape(k)}={_escape(strings[k])}" for k in sorted(keys))
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
       recommended, since the canonical flat keys like "metric.p" are derived for you) or as an
       iterable of canonical flat key strings. All candidate specs must share this key set (the
       usual grid-of-jobs situation).
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
        `logged_run(index=...)`). With `experiment_names=None`, mlflow.search_runs searches
        only the currently-active experiment (i.e. the last `mlflow.set_experiment`).
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
    "select_spec",
    "RunIndex",
]
