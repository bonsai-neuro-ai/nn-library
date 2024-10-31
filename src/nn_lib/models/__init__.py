from .lit_classifier import LitClassifier
from jsonargparse import ArgumentParser
from typing import Type
from torch import nn
from torchvision.models import get_model as tv_get_model, get_model_weights as tv_get_weights
from torchvision.transforms._presets import SemanticSegmentation


# def add_parser(
#     parser: ArgumentParser,
#     key: str = "model",
#     baseclass: Type[nn.Module] = GraphModule,
#     instantiate: bool = True,
# ):
#     parser.add_subclass_arguments(baseclass=baseclass, nested_key=key, instantiate=instantiate)
#
#     # Create/update 'metafields' attribute on the parser
#     if hasattr(baseclass, "__metafields__"):
#         meta = getattr(parser, "metafields", {})
#         meta.update({key: baseclass.__metafields__})
#         setattr(parser, "metafields", meta)


def get_pretrained_model(name: str) -> nn.Module:
    weights = tv_get_weights(name).DEFAULT
    return tv_get_model(name, weights=weights)


def get_default_transforms(name: str):
    weights = tv_get_weights(name).DEFAULT
    trans = weights.transforms()

    # TODO - is this bugfix sensible? Some models have a single resize size which has been causing
    #  problems with jagged tensor sizes.
    if isinstance(trans, SemanticSegmentation):
        if len(trans.resize_size) == 1:
            trans.resize_size = [trans.resize_size[0]] * 2
    return trans


__all__ = [
    "LitClassifier",
    # "ResNet",  # TODO - fix resnet after refactor
    # "add_parser",  # TODO - fix parser after refactor
    "get_pretrained_model",
    "get_default_transforms",
]
