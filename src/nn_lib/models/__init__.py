from torch import nn
from torch.fx import Graph, symbolic_trace
from torchvision.models import get_model as tv_get_model, get_model_weights as tv_get_weights

from nn_lib.utils import deprecated
from .graph_module_plus import GraphModulePlus
from .lit_classifier import LitClassifier


@deprecated("Use the GraphModelPlus interface instead.")
def get_model_graph(name: str, squash: bool = False) -> Graph:
    from .graph_utils import squash_all_conv_batchnorm_pairs
    graph_module = symbolic_trace(tv_get_model(name))
    if squash:
        graph_module = squash_all_conv_batchnorm_pairs(graph_module)
    return graph_module.graph


def get_pretrained_model(name: str) -> nn.Module:
    weights = tv_get_weights(name).DEFAULT
    return tv_get_model(name, weights=weights)


__all__ = [
    "LitClassifier",
    "GraphModulePlus",
    "get_pretrained_model",
]
