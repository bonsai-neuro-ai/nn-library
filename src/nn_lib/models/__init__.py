from .lit_classifier import LitClassifier
from jsonargparse import ArgumentParser
from typing import Type, assert_never
from torch import nn
from torchvision.models import get_model as tv_get_model, get_model_weights as tv_get_weights
from torchvision.transforms._presets import (
    ObjectDetection,
    ImageClassification,
    VideoClassification,
    SemanticSegmentation,
    OpticalFlow,
)


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


def get_default_transforms(name: str, max_size: int = 256):
    weights = tv_get_weights(name).DEFAULT
    trans = weights.transforms()

    match trans:
        case SemanticSegmentation() | ImageClassification() | VideoClassification():
            # TODO - is this bugfix sensible? Some models have a single resize size which has
            #  been causing problems with jagged tensor sizes.
            if len(trans.resize_size) == 1:
                trans.resize_size = [trans.resize_size[0]] * 2
            trans.resize_size = [min(max_size, size) for size in trans.resize_size]
        case ObjectDetection() | SemanticSegmentation() | OpticalFlow():
            pass
        case _:
            assert_never(trans)
    return trans


__all__ = [
    "LitClassifier",
    # "ResNet",  # TODO - fix resnet after refactor
    # "add_parser",  # TODO - fix parser after refactor
    "get_pretrained_model",
    "get_default_transforms",
]
