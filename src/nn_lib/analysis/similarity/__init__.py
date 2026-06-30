"""Neural representation similarity metrics: linear CKA, HSIC, and shape-distance variants."""

from .cka import LinearCKA, HSICEstimator
from .shape_distance import ShapeDistance, CrossValidatedShapeDistance
