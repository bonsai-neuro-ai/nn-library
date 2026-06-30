"""PyTorch model definitions, custom layers, and utilities for loading/reconfiguring models."""

from torch import nn
from torchvision.models import get_model as tv_get_model, get_model_weights as tv_get_weights

from .fancy_layers import *
from .graph_module_plus import GraphModulePlus
from .sparse_auto_encoder import *


def get_pretrained_model(name: str) -> nn.Module:
    """Load a torchvision model by name (see `torchvision.models.list_models()`) with its
    default pretrained weights."""
    weights = tv_get_weights(name).DEFAULT
    return tv_get_model(name, weights=weights)
