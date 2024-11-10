from .mnist import MNISTDataModule
from .cifar import CIFAR10DataModule, CIFAR100DataModule
from .imagenet import ImageNetDataModule
from .coco_semantic_segmentation import CocoDetectionDataModule
from .base import TorchvisionDataModuleBase
from .transforms import get_tv_default_transforms
from jsonargparse import ArgumentParser


def add_parser(parser: ArgumentParser, key: str = "data", link_env: bool = True):
    parser.add_subclass_arguments(baseclass=TorchvisionDataModuleBase, nested_key=key)
    if link_env:
        parser.link_arguments("env.data_root", "data.init_args.root_dir", apply_on="parse")

    # Create/update 'metafields' attribute on the parser
    meta = getattr(parser, "metafields", {})
    meta.update({key: TorchvisionDataModuleBase.__metafields__})
    setattr(parser, "metafields", meta)
