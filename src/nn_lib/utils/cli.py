from argparse import Namespace
from typing import (
    assert_never,
    Optional,
    Iterable,
    Union,
    Any,
    get_args,
    Callable,
    Generator,
    Mapping,
)

from jsonargparse import Namespace as JSONNamespace, strip_meta

ParamsLike = Union[dict, Namespace, JSONNamespace]
NestedKey = Union[None, str, list[str], dict[str, "NestedKey"]]


def _params_to_dict_no_recursion(params: ParamsLike) -> dict:
    """Cast any ParamsLike object to a dictionary. Does not recurse, and returned dict should not be
    modified (it may be a view of the original object).
    """
    match params:
        case JSONNamespace():
            return strip_meta(params).as_dict()
        case Namespace():
            return vars(params)
        case dict():
            return params
        case _:
            assert_never(params)


def _is_key_skipped(k: str, skip_spec: NestedKey) -> tuple[bool, NestedKey]:
    """Type-agnostic helper to determine, inside _iter_flatten_params, if a key should be skipped.

    Returns a tuple of (should_skip, skip_spec). If should_skip is True, the key is skipped. If
    should_skip is False, the key is not skipped, and the returned skip_spec should be used in
    recursive calls to check if sub-keys should be skipped.
    """
    match skip_spec:
        case None:
            # if the skip_spec is None, we don't skip anything
            return False, None
        case str():
            # if the skip_spec is a string, check if it matches the key
            return k == skip_spec, None
        case list():
            # if the skip_spec is an iterable, check if the key is in it
            return k in skip_spec, None
        case dict():
            # if the skip_spec is a mapping, it could either be like {k: v} or {k: {k2: v2}}. In
            # the former case, we skip any key that is in the mapping, and in the latter case we
            # leave it to the recursive call to check if the key is in the mapping
            if k not in skip_spec:
                return False, None
            elif isinstance(skip_spec[k], dict):
                # Here, we have some nested skip_spec, so we say that key 'k' is not skipped,
                # leaving it to other recursive calls to check sub-keys like k2.
                return False, skip_spec[k]
            else:
                # Here, we say the key 'k' is skipped if the skip_spec[k] maps to something truthy.
                return bool(skip_spec[k]), None
        case _:
            assert_never(skip_spec)


def _iter_flatten_params(
    d: ParamsLike,
    join_op: Callable,
    prefix: tuple = tuple(),
    skip_keys: Optional[NestedKey] = None,
) -> Generator[tuple[str, Any], None, None]:
    """Iterate a nested dict/params in order, yielding (k1k2k3, v) from a namespace like
    Namespace(a=1, b=Namespace(c=2, d=3)) or a dict like {k1: {k2: {k3: v}}}. Uses the given join_op
    to join keys together. In this example, join_op(k1, k2, k3) should return k1k2k3.

    'skip_keys' can specify fields to skip OR some field inside another field to skip.
    """
    d = _params_to_dict_no_recursion(d)
    for k, v in d.items():
        should_skip, sub_skip_spec = _is_key_skipped(k, skip_keys)
        if should_skip:
            continue
        new_prefix = prefix + (k,)
        if isinstance(v, get_args(ParamsLike)):
            yield from _iter_flatten_params(v, join_op, new_prefix, sub_skip_spec)
        else:
            joined_key = join_op(new_prefix)
            yield joined_key, v


def flatten_params(params: ParamsLike, ignore: NestedKey = None) -> dict:
    """Flatten the given parameters. If the parameters are a Namespace, they will be converted to
    a dictionary first. Nested parameters are flattened recursively.
    """
    return dict(_iter_flatten_params(params, join_op="/".join, skip_keys=ignore))


__all__ = [
    "flatten_params",
    "ParamsLike",
    "NestedKey",
]