from .lit_classifier import LitClassifier
from .resnet import ResNet
from .graph_module import GraphModule
from jsonargparse import ArgumentParser


def add_parser(parser: ArgumentParser, key: str = "model"):
    parser.add_subclass_arguments(baseclass=LitClassifier, nested_key=key)

    # Create/update 'metafields' attribute on the parser
    meta = getattr(parser, "metafields", {})
    meta.update({key: LitClassifier.__metafields__})
    setattr(parser, "metafields", meta)
