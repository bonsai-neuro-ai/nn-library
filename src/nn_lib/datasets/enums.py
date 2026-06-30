from enum import Enum, auto


class TorchvisionDatasetType(Enum):
    """Task type of a `TorchvisionDataModuleBase` subclass; determines whether it needs a joint
    input/target `transforms` callable (see `TorchvisionDataModuleBase.requires_target_transform`)
    and how `get_default_transforms_v2` builds its default transform."""

    OBJECT_DETECTION = auto()
    IMAGE_CLASSIFICATION = auto()
    VIDEO_CLASSIFICATION = auto()
    SEMANTIC_SEGMENTATION = auto()
    OPTICAL_FLOW = auto()
