import pydot
import torch
from torch import nn
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union


# Type definitions for graph-based model construction
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
# Once called, the model might produce a Tensor or a dict of named tensors
OutputType = Union[torch.Tensor, Dict[str, torch.Tensor]]


class Add:
    def __call__(self, x, y):
        return x + y


class Identity:
    def __call__(self, x):
        return x


def graph2dot(network_graph: GraphType) -> pydot.Graph:
    edges = []
    for layer, (_, parents) in network_graph.items():
        for pa in parents:
            edges.append((pa, layer))
    return pydot.graph_from_edges(edges, directed=True)


def squash_conv_batchnorm(conv_layer: nn.Conv2d, bn_layer: nn.BatchNorm2d) -> nn.Conv2d:
    """Construct a single conv2d layer which behaves equivalently to batchnorm(conv2d(x)) by
    absorbing the batchnorm parameters into the convolution weights and bias.
    """
    # Thanks to https://discuss.pytorch.org/t/how-to-absorb-batch-norm-layer-weights-into-convolution-layer-weights/16412/5
    mean = bn_layer.running_mean
    var_sqrt = torch.sqrt(bn_layer.running_var + bn_layer.eps)
    gamma = bn_layer.weight
    beta = bn_layer.bias
    if conv_layer.bias is not None:
        prev_conv_bias = conv_layer.bias
    else:
        prev_conv_bias = mean.new_zeros(mean.shape)
    new_conv_weight = conv_layer.weight * (gamma / var_sqrt).reshape(
        [conv_layer.out_channels, 1, 1, 1]
    )
    new_conv_bias = (prev_conv_bias - mean) / var_sqrt * gamma + beta
    fused_conv = nn.Conv2d(
        in_channels=conv_layer.in_channels,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=True,
    )
    fused_conv.weight = nn.Parameter(new_conv_weight)
    fused_conv.bias = nn.Parameter(new_conv_bias)
    return fused_conv
