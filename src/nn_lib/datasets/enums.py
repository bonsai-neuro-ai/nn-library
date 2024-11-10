from enum import Enum, auto


class TorchvisionDatasetType(Enum):
    OBJECT_DETECTION = auto()
    IMAGE_CLASSIFICATION = auto()
    VIDEO_CLASSIFICATION = auto()
    SEMANTIC_SEGMENTATION = auto()
    OPTICAL_FLOW = auto()
