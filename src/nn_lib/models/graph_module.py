import torch
import torch.nn as nn
from typing import Dict, Tuple, Callable, Union, Any, Iterable, List, Optional, Literal, Generator
import warnings
from nn_lib.utils import iter_flatten_dict
from collections import deque, defaultdict


# An 'Operation' is essentially a callable object that behaves like a nn.Module. Use None for
# input values.
OpType = Union[None, Callable, nn.Module, "ModelType"]
# Specify inputs by name (string), relative index (negative integer), or a list of either
InputType = Union[str, int]
# An 'Operation' can be specified as a tuple of (op, input), where input can be a string, int, or
# list of them
OpWithMaybeInputs = Union[OpType, Tuple[OpType, InputType], Tuple[OpType, Iterable[InputType]]]
# A model is a dict of named operations. Operations can themselves contain Models, so this allows
# for model to be nested.
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

    def __init__(
        self,
        architecture: ModelType,
        inputs: Optional[Iterable[str]] = None,
        outputs: Optional[Iterable[str]] = None,
    ):
        super(GraphModule, self).__init__()
        # Convert ModelType specification of an architecture into 'flatter' GraphType
        self.graph = model2graph(architecture)

        # Add all {name: operation} modules as a ModuleDict so that they appear in self.parameters
        self.module_dict = nn.ModuleDict(
            {path: op for path, (op, inpts) in self.graph.items() if isinstance(op, nn.Module)}
        )
        # Save the input name(s) for use in forward(). Defaults to global INPUT_LAYER. Also track
        # output names (we always return all layers; setting the output layer names is for
        # bookkeeping and does not change the model behavior). Defaults to the last key in the
        # architecture dict.
        self.inputs = list(inputs) if inputs is not None else [INPUT_LAYER]
        self.outputs = list(outputs) if outputs is not None else list(architecture.keys())[-1:]

        # Check that all inputs and outputs are present in the graph
        for input_name in self.inputs:
            if input_name not in self.graph:
                warnings.warn(
                    f"No '{input_name}' node in the graph! "
                    "Calls to forward() will need to specify inputs by name!"
                )
        for output_name in self.outputs:
            if output_name not in self.graph:
                warnings.warn(
                    f"No '{output_name}' node in the graph, but it is named as an output!"
                )

    def reorder_graph(self):
        """Ensure that graph traversal using self.graph.items() is in topological order."""
        new_graph = {}
        for layer_name in self._traverse_sub_graph(self.inputs, self.layer_names):
            new_graph[layer_name] = self.graph[layer_name]
        self.graph = new_graph

    @property
    def layer_names(self):
        return list(self.graph.keys())

    def architecture(self, root: Optional[str] = None, sep="/") -> ModelType:
        """Return the architecture of this model as a ModelType specification."""
        # All input paths will be made relative to a new root if given
        update_path = lambda inpt: (
            root + sep + inpt if (root is not None and isinstance(inpt, str)) else inpt
        )
        return {
            layer_name: (op, [update_path(inpt) for inpt in inpts])
            for layer_name, (op, inpts) in self.graph.items()
        }

    def get_op(self, layer_name: str) -> nn.Module:
        """Return the nn.Module operation for the given layer name."""
        return self.graph[layer_name][0]

    def __getitem__(self, item):
        """Call model[layer_name] to get the operation for that layer."""
        return self.get_op(item)

    def _handle_initial_inputs(
        self, initial_inputs: Union[torch.Tensor, Iterable[torch.Tensor], dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        if isinstance(initial_inputs, dict):
            missing_inputs = set(self.inputs) - set(initial_inputs.keys())
            if missing_inputs:
                raise ValueError(
                    f"Required inputs {missing_inputs} not found in the initial_inputs dict!"
                )
            # Make a copy of the dict (but note this points to the same tensors)
            return dict(initial_inputs.items())
        elif torch.is_tensor(initial_inputs):
            if len(self.inputs) == 1:
                return {self.inputs[0]: initial_inputs}
            else:
                raise ValueError(
                    f"Multiple inputs specified in the graph, but only one tensor provided!"
                )
        else:
            return self._handle_initial_inputs(dict(zip(self.inputs, initial_inputs)))

    def forward(
        self,
        initial_inputs: Union[torch.Tensor, Iterable[torch.Tensor], dict[str, torch.Tensor]],
        named_outputs: Optional[Iterable[str]] = None,
        warn_if_missing: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run the graph forward, starting with the given initial inputs. If named_outputs is
        specified, only compute those outputs and return them as a dict. If warn_if_missing is
        True, a warning will be raised if any layer's inputs are not available in the output dict.
        """
        layer_activations = self._handle_initial_inputs(initial_inputs)

        if named_outputs is not None:
            # If named_outputs is specified, only compute those outputs and return them as a dict.
            # This also ensures no unnecessary layers are computed.
            traversal_order = self._traverse_sub_graph(layer_activations.keys(), named_outputs)
        else:
            traversal_order = self.graph.keys()

        # Run inputs forward and compute everything we can, or just compute 'outputs' if
        # supplied. Returned dict contains all inputs, hidden activations, and outputs, keyed by
        # their 'path' name in the graph
        for layer_name in traversal_order:
            op, layer_inputs = self.graph[layer_name]
            if layer_name in layer_activations:
                continue
            elif all(inpt in layer_activations for inpt in layer_inputs):
                if op is None:
                    # Any 'None' operations are identity operations. Must have only a single input.
                    assert len(layer_inputs) == 1, "'None' operations must have a single input!"
                    layer_activations[layer_name] = layer_activations[layer_inputs[0]]
                else:
                    # Do the forward pass for this layer
                    layer_activations[layer_name] = op(
                        *(layer_activations[inpt] for inpt in layer_inputs)
                    )
            elif warn_if_missing:
                warnings.warn(
                    f"Skipping {layer_name} because inputs are not available! "
                    f"Calling reorder_graph() could fix this if it's a traversal order issue."
                )
        return layer_activations

    def _traverse_sub_graph(
        self, inputs: Iterable[str], outputs: Iterable[str]
    ) -> Generator[str, None, None]:
        """Walk self.graph with traversal order to build outputs from inputs. Yields layer names."""
        inputs, outputs = list(inputs), list(outputs)
        for layer_name in inputs + outputs:
            if layer_name not in self.graph:
                raise ValueError(f"Layer {layer_name} not found in the graph!")

        visited, leaves = set(), set(inputs)

        def visit(layer_name: str):
            """Recursively visit a layer's inputs first, then the layer itself."""
            nonlocal visited, leaves

            # Don't visit a layer twice
            if layer_name in visited:
                return
            visited.add(layer_name)

            if layer_name in leaves:
                # No recursion from leaves
                yield layer_name
            else:
                # Recurse to inputs first...
                _, layer_inputs = self.graph[layer_name]
                for inpt in layer_inputs:
                    yield from visit(inpt)
                # ...then visit the layer itself
                yield layer_name

        for output in outputs:
            yield from visit(output)

    def sub_model(self, inputs: list[str], outputs: list[str]) -> "GraphModule":
        """Return a new GraphModule containing the sub-graph mapping from the given inputs to the
        given outputs.
        """
        sub_graph: ModelType = {
            name: self.graph[name] for name in self._traverse_sub_graph(inputs, outputs)
        }
        for inpt in inputs:
            sub_graph[inpt] = None
        return GraphModule(sub_graph, inputs=inputs, outputs=outputs)


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
