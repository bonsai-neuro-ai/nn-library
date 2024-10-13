from .lit_classifier import LitClassifier
from .resnet import ResNet
from .graph_module import GraphModule
from jsonargparse import ArgumentParser
from typing import Type
from torch import nn


def add_parser(
    parser: ArgumentParser,
    key: str = "model",
    baseclass: Type[nn.Module] = GraphModule,
):
    parser.add_subclass_arguments(baseclass=baseclass, nested_key=key)

    # Create/update 'metafields' attribute on the parser
    if hasattr(baseclass, "__metafields__"):
        meta = getattr(parser, "metafields", {})
        meta.update({key: baseclass.__metafields__})
        setattr(parser, "metafields", meta)


__all__ = [
    "LitClassifier",
    "ResNet",
    "GraphModule",
    "add_parser",
]
