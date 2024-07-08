import torch
import torch.nn as nn
from typing import Dict, Tuple, Callable, Union, Any, Iterable, List, Optional
import warnings
from nn_lib.utils import iter_flatten_dict

# An 'Operation' is essentially a callable object that behaves like a nn.Module. Use None for
# input values.
OpType = Union[None, Callable, nn.Module, "ModelType"]
# Specify inputs by name (string), relative index (negative integer), or a list of either
InputType = Union[str, int]
# An 'Operation' can be specified as a tuple of (op, input), where input can be a string, int, or
# list of them
OpWithMaybeInputs = Union[OpType, Tuple[OpType, InputType], Tuple[OpType, Iterable[InputType]]]
# A model is a dict of named operations. Operations can themselves contain Models, so this allows
# for models to be nested.
ModelType = Dict[str, OpWithMaybeInputs]
# A Graph is a map like {node: (edge, [parents])}, where edge can be Any. Here, edges will be
# Callables
GraphType = Dict[str, Tuple[Any, List[str]]]


INPUT_LAYER = "input"


class GraphModule(nn.Module):
    """A GraphModule is a PyTorch Module specified by a graph of named operations. Implementation
    draws inspiration from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py

    The constructor takes a single 'architecture' argument, which is a nested dict of the form
    {
        'layer1': {'step1': op1, 'step2': (op2, 'step1')},
        'layer2': {'step1': op3, 'step2': (op4, 'step1')}
    }
    """

    def __init__(self, architecture: ModelType):
        super(GraphModule, self).__init__()
        # Convert ModelType specification of an architecture into 'flatter' GraphType
        self.graph = model2graph(architecture)
        if INPUT_LAYER not in self.graph:
            warnings.warn(
                f"No '{INPUT_LAYER}' node in the graph! "
                "Calls to forward() will need to specify inputs by name!"
            )
        # Add all {name: operation} modules as a ModuleDict so that they appear in self.parameters
        self.module_dict = nn.ModuleDict(
            {path: op for path, (op, inpts) in self.graph.items() if isinstance(op, nn.Module)}
        )

    @property
    def layer_names(self):
        return list(self.graph.keys())

    def forward(
        self,
        initial_inputs: Union[torch.Tensor, dict],
        named_outputs: Optional[Iterable[str]] = None,
        warn_if_missing: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(initial_inputs, dict):
            initial_inputs = {"input": initial_inputs}
        if named_outputs is not None:
            # If named_outputs is specified, only compute those outputs and return them as a dict.
            # TODO - implement efficient graph traversal to avoid computing unnecessary outputs.
            out = self.forward(initial_inputs, None, warn_if_missing)
            return {name: out[name] for name in named_outputs}

        # Run inputs forward and compute everything we can, or just compute 'outputs' if
        # supplied. Returned dict contains all inputs, hidden activations, and outputs, keyed by
        # their 'path' name in the graph
        out = dict(initial_inputs)
        for layer_name, (op, layer_inputs) in self.graph.items():
            if layer_name in initial_inputs:
                continue
            elif all(inpt in out for inpt in layer_inputs):
                out[layer_name] = op(*(out[inpt] for inpt in layer_inputs))
            elif warn_if_missing:
                warnings.warn(f"Skipping {layer_name} because inputs are not available!")
        return out


#####################
## graph building ##
#####################


def _canonicalize_input(op: OpWithMaybeInputs) -> Tuple[OpType, List[InputType]]:
    """The input to an operation can be specified in a few ways (type OpWithMaybeInputs). This
    function ensures a canonical form of (op, inputs) where inputs is a list of strings or ints.
    """
    if isinstance(op, tuple):
        op, inpts = op
        if isinstance(inpts, str):
            return op, [inpts]
        elif isinstance(inpts, int):
            return op, [inpts]
        elif isinstance(inpts, list):
            return op, inpts
        else:
            raise ValueError(f"Cannot parse (op, inputs): {(op, inpts)}")
    else:
        # If no input is explicitly specified, assume a single input pointing to the previous
        # layer's output
        return op, [-1]


def _normpath(path, sep="/"):
    # simplified os.path.normpath
    parts = []
    for p in path.split(sep):
        if p == "..":
            parts.pop()
        elif p.startswith(sep):
            parts = [p]
        else:
            parts.append(p)
    return sep.join(parts)


def model2graph(model: ModelType, sep="/") -> GraphType:
    """Convert a nested dict of operations into a flat graph, where each operation is keyed by a
    unique path and may specify inputs using relative names or relative indices. The graph is a
    dict of {node: (edge, [parents])} where edge is a callable object (a nn.Module) and parents
    is a list of strings or integers specifying a relative path to inputs. The graph is
    constructed by flattening the nested dict and resolving relative paths or indices in the inputs.

    Example:

        model = {'layer1': {'step1': op1, 'step2': (op2, 'step1')}}

    will be converted to:

        graph = {
            'layer1/step1': (op1, ['input']),
            'layer1/step2': (op2, ['layer1/step1'])
        }
    """
    # Note that we don't convert this into a dict here in case of name conflicts.
    flattened_layers = [
        (joined_path, _canonicalize_input(op))
        for joined_path, op in iter_flatten_dict(model, join_op=sep.join)
    ]

    # Resolve input references. In the example above, op1 has no named input, so it will use
    # whatever the previous layer was, and 'step2' refers locally to its sibling 'layer1/step1'.
    # The first layer will use the INPUT_LAYER key as its input by default.
    graph = {}
    for idx, (path, (op, inpts)) in enumerate(flattened_layers):
        # Iterate over inputs and resolve any relative paths or indices
        if op is not None:
            for i, inpt in enumerate(inpts):
                if isinstance(inpt, int):
                    # 'inpt' is an int like -1, referring to some number of layers back. Get that
                    # layer's name as a string
                    inpts[i] = flattened_layers[idx + inpt][0]
                elif isinstance(inpt, str) and inpt not in graph:
                    # 'inpt' is a string specifying a particular layer, but it's not a key in
                    # 'graph' (not an absolute path)
                    inpts[i] = _normpath(sep.join([path, "..", inpt]), sep=sep)
                # Sanity-check
                assert inpts[i] in graph, (
                    f"While building graph, input to {path} includes {inpts[i]}, "
                    f"but keys so far are {list(graph.keys())}"
                )

        if path in graph:
            raise ValueError(f"Duplicate path {path} in the graph!")

        # Add this op to the graph using its absolute paths
        graph[path] = (op, inpts)

    return graph


__all__ = ["model2graph", "GraphModule"]
