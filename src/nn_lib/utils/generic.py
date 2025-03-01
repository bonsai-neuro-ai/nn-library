import inspect
import itertools
import warnings
from typing import Optional, Callable, Generator, Tuple, Any, TypeVar, Iterable, Union

import torch
from tqdm.auto import tqdm
from functools import wraps

try:
    from warnings import deprecated
except ImportError:

    def deprecated(reason):
        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                warnings.warn(reason, DeprecationWarning, stacklevel=2)
                return func(*args, **kwargs)

            return wrapped

        return decorator


T = TypeVar("T")
K = TypeVar("K")
J = TypeVar("J")


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


def vmap_debug(fn, in_axes=None, out_axes=None, progbar: bool = False) -> Callable:
    """Debugging version of torch.func.vmap that does things in an explicit python loop"""

    def iter_axis(tensor: torch.Tensor, axis: int) -> Generator[torch.Tensor, None, None]:
        if axis is None:
            yield tensor
        else:
            for i in range(tensor.size(axis)):
                yield tensor.select(axis, i)

    def wrapped(*args):
        nonlocal in_axes, out_axes
        if in_axes is None:
            in_axes = tuple(range(len(args)))
        out_collection_shape = tuple(
            arg.size(axis) for arg, axis in zip(args, in_axes) if axis is not None
        )
        if out_axes is None:
            out_axes = tuple(range(len(out_collection_shape)))

        if not isinstance(out_axes, tuple):
            out_axes = tuple(out_axes)

        assert len(args) == len(in_axes), "Need one input per input axis"
        assert len(out_collection_shape) == len(out_axes), "Need one output axis per input axis"

        # Create an iterator for all combinations of args. Non-mapped args are repeated.
        results = []
        iterator = itertools.product(*[iter_axis(arg, axis) for arg, axis in zip(args, in_axes)])
        if progbar:
            iterator = tqdm(
                iterator, total=torch.tensor(out_collection_shape).prod().item(), leave=False
            )
        for in_args in iterator:
            results.append(fn(*in_args))
        shape_single_output = results[-1].shape
        return (
            torch.stack(results, dim=0)
            .reshape(out_collection_shape + shape_single_output)
            .permute(out_axes + tuple(i + len(out_axes) for i in range(len(shape_single_output))))
        )

    return wrapped


__all__ = [
    "deprecated",
    "iter_flatten_dict",
    "supersedes",
    "vmap_debug",
]
