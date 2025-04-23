import itertools
import warnings
from functools import wraps
from typing import Callable, Generator, TypeVar

import torch
from tqdm.auto import tqdm

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


# It's understood but not enforced that ChildType is a subclass of ParentType
ParentType = TypeVar("ParentType")
ChildType = TypeVar("ChildType")


def supersedes(parent_class: ParentType) -> Callable[[ChildType], ChildType]:
    """Decorator to place on some subclass to mark it as the new version of the parent class.
    All instances of the parent class, globally, will be replaced with instances of the new class,
    thanks to our ability to mess with the __new__ method of the parent class.

    Example:

        class Foo(object):
            def __init__(self, x):
                self.x = x
            def foo(self):
                print("foo", self.x)

        def foo_factory(x):
            return Foo(x)

        @supersedes(Foo)
        class BetterFoo(Foo):
            def __init__(self, x):
                super().__init__(x + 1)  # BetterFoo is better because x+1 is better than x

            # BetterFoo also defines some new methods
            def bar(self):
                print("bar", self.x)

        f = foo_factory(10)
        f.foo()  # prints 11
        f.bar()  # type checker is unhappy, but this works!
        print(type(f))  # prints BetterFoo

        f = BetterFoo(10)
        f.foo()  # prints 11
        f.bar()  # prints bar 11, and the type checker is happy
    """

    def decorator(child_class: ChildType) -> ChildType:
        assert issubclass(child_class, parent_class)  # type: ignore
        og_new = parent_class.__new__

        def new_new(cls, *args, **kwargs):
            if cls is parent_class:
                cls = child_class
            # Sometimes __new__ takes additional args, sometimes not. We'll just try with args first
            # and fall back on the no-args case if that fails.
            try:
                return og_new(cls, *args, **kwargs)
            except TypeError:
                return og_new(cls)

        parent_class.__new__ = classmethod(new_new)
        return child_class

    return decorator


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
    "supersedes",
    "vmap_debug",
]
