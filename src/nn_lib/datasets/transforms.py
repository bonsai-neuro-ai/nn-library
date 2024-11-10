import torch
from torchvision.transforms._presets import (
    ImageClassification,
    ObjectDetection,
    OpticalFlow,
    SemanticSegmentation,
    VideoClassification,
)
from torchvision.transforms import v2
from torchvision.models import get_model_weights as tv_get_weights
from typing import Optional, assert_never, TYPE_CHECKING
from nn_lib.datasets.enums import TorchvisionDatasetType
from typing import Union

if TYPE_CHECKING:
    from nn_lib.datasets.base import TorchvisionDataModuleBase


TVDefaultTransform = Union[
    ImageClassification,
    ObjectDetection,
    OpticalFlow,
    SemanticSegmentation,
    VideoClassification,
]
V2Transform = v2.Transform

SizeArg = Union[int, tuple[int], list[int]]


def _handle_size(size: SizeArg) -> list[int]:
    """Handle 'size' arguments, returning a canonical [height, width] list format."""
    if isinstance(size, int):
        return [size, size]
    else:
        if len(size) > 2:
            raise ValueError("Size must be a tuple or list containing 1 or 2 elements.")
        # Indexing with 0 and -1 lets us handle cases where size tuples are length 1 OR length 2.
        return [size[0], size[-1]]


def tv_preset_transform_to_v2(
    tv_transform: TVDefaultTransform,
    max_size: Optional[SizeArg] = None,
    ignore_targets: bool = False,
) -> v2.Transform:
    """Convert any transform coming from torchvision's 'presets' (e.g. default model weights'
    transforms() object) to a v2 transform. This is useful when we want to use the same transform
    for both images and labels.

    :param tv_transform: A transform object from torchvision's 'presets' module.
    :param max_size: An optional (max_height, max_width) tuple which may limit image sizes.
    :param ignore_targets: If True, don't do any target (label) transformations. This is useful for
        instance if we want to run a segmentation model on a classification dataset where there are
        no segmentation mask labels.
    :return: A v2 transform object with the same behavior which will also work on labels.
    """
    # TODO: add a 'random_augmentation' flag to this function which changes behavior depending
    #  on whether we're in training or testing mode.
    sz = _handle_size(tv_transform.resize_size)
    if max_size is not None:
        max_sz = _handle_size(max_size)
        sz = [min(sz[0], max_sz[0]), min(sz[1], max_sz[1])]

    match tv_transform:
        case ImageClassification():
            return v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(
                        sz,
                        interpolation=tv_transform.interpolation,
                        antialias=tv_transform.antialias,
                    ),
                    v2.CenterCrop(tv_transform.crop_size),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(tv_transform.mean, tv_transform.std),
                ]
            )
        case SemanticSegmentation():
            return v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(
                        sz,
                        interpolation=tv_transform.interpolation,
                        antialias=tv_transform.antialias,
                    ),
                    v2.SanitizeBoundingBoxes() if not ignore_targets else v2.Identity(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(tv_transform.mean, tv_transform.std),
                ]
            )
        case ObjectDetection():
            raise NotImplementedError("ObjectDetection transforms not implemented")
        case OpticalFlow():
            raise NotImplementedError("OpticalFlow transforms not implemented")
        case VideoClassification():
            raise NotImplementedError("VideoClassification transforms not implemented")
        case _:
            assert_never(tv_transform)


def get_tv_default_transforms(
    model_name: str, max_size: Optional[SizeArg] = None, ignore_targets: bool = False
) -> v2.Transform:
    """Get the default transforms for a given torchvision model."""
    weights = tv_get_weights(model_name).DEFAULT
    return tv_preset_transform_to_v2(
        weights.transforms(), max_size=max_size, ignore_targets=ignore_targets
    )


def get_default_transforms_v2(
    datamodule: "TorchvisionDataModuleBase", ignore_targets: bool = False
) -> v2.Transform:
    """Get the default transforms for a given TorchvisionDataModuleBase subclass. This is *our*
    defaults, not torchvision's defaults. See `get_tv_default_transforms` for torchvision's, and
    `tv_preset_transform_to_v2` for converting them to v2 transforms.
    """
    meta = datamodule.metadata
    match datamodule.type:
        case TorchvisionDatasetType.IMAGE_CLASSIFICATION:
            return tv_preset_transform_to_v2(
                ImageClassification(
                    crop_size=datamodule._default_shape[1],
                    resize_size=datamodule._default_shape[1],
                    mean=meta["mean"],
                    std=meta["std"],
                ),
                ignore_targets=ignore_targets,
            )
        case TorchvisionDatasetType.SEMANTIC_SEGMENTATION:
            return tv_preset_transform_to_v2(
                SemanticSegmentation(
                    resize_size=datamodule._default_shape[1],
                    mean=meta["mean"],
                    std=meta["std"],
                ),
                ignore_targets=ignore_targets,
            )
        case (
            TorchvisionDatasetType.OBJECT_DETECTION
            | TorchvisionDatasetType.VIDEO_CLASSIFICATION
            | TorchvisionDatasetType.OPTICAL_FLOW
        ):
            raise NotImplementedError(
                f"Default transforms for {datamodule.type} not implemented yet"
            )
        case _:
            assert_never(datamodule.type)


__all__ = [
    "get_default_transforms_v2",
    "get_tv_default_transforms",
    "tv_preset_transform_to_v2",
    "TVDefaultTransform",
    "V2Transform",
]
