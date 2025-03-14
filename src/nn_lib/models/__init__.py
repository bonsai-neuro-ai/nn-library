from torch import nn
from torchvision.models import get_model as tv_get_model, get_model_weights as tv_get_weights

from .graph_module_plus import GraphModulePlus
from .lit_classifier import LitClassifier
from .fancy_layers import *
from .sparse_auto_encoder import *


def get_pretrained_model(name: str) -> nn.Module:
    weights = tv_get_weights(name).DEFAULT
    return tv_get_model(name, weights=weights)


__all__ = [
    "LitClassifier",
    "GraphModulePlus",
    "get_pretrained_model",
]
