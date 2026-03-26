"""OVD model wrappers."""

from .base_ovd import OVDConfig, DetectionResult, BoundingBox, BaseOVDModel
from .yolo_world import YOLOEModel

__all__ = [
    "OVDConfig",
    "DetectionResult",
    "BoundingBox",
    "BaseOVDModel",
    "YOLOEModel",
]
