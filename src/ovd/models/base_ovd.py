"""Base OVD model interface and data structures.

This module defines the interface for Open-Vocabulary Detection models
and the data structures used for detection results.
"""

from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class OVDConfig:
    """Configuration for OVD model inference.

    Attributes:
        model_name: Name or path of the model file
        conf_threshold: Confidence threshold for detections (0-1)
        iou_threshold: IoU threshold for NMS (0-1)
        max_detections: Maximum number of detections per image
        device: Device to run inference on ('cpu', 'cuda:0', etc.)
    """

    model_name: str = "yoloe-26x-seg.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    device: str = "cpu"


@dataclass(frozen=True)
class BoundingBox:
    """A single bounding box detection.

    【重要】字段含义已优化：
    - x1: 归一化的中心 x 坐标 (x_center, 0-1)
    - y1: 归一化的中心 y 坐标 (y_center, 0-1)
    - x2: 归一化的宽度 (width, 0-1)
    - y2: 归一化的高度 (height, 0-1)

    这样可以直接存储 YOLO 格式的归一化坐标，上层直接使用无需转换。

    Attributes:
        x1: Normalized center x coordinate (0-1)
        y1: Normalized center y coordinate (0-1)
        x2: Normalized width (0-1)
        y2: Normalized height (0-1)
        class_id: Class index (0-based)
        confidence: Detection confidence score (0-1)
        class_name: Human-readable class name
    """

    x1: float  # x_center (normalized)
    y1: float  # y_center (normalized)
    x2: float  # width (normalized)
    y2: float  # height (normalized)
    class_id: int
    confidence: float
    class_name: str

    @property
    def x_center(self) -> float:
        """Normalized center x coordinate."""
        return self.x1

    @property
    def y_center(self) -> float:
        """Normalized center y coordinate."""
        return self.y1

    @property
    def width(self) -> float:
        """Normalized width."""
        return self.x2

    @property
    def height(self) -> float:
        """Normalized height."""
        return self.y2

    @property
    def area(self) -> float:
        """Normalized area."""
        return self.width * self.height


@dataclass
class DetectionResult:
    """Result of OVD inference on a single image.

    Attributes:
        image_path: Path to the input image
        boxes: List of detected bounding boxes (with normalized YOLO format)
        inference_time: Time taken for inference in seconds
    """

    image_path: str
    boxes: List[BoundingBox]
    inference_time: float


class BaseOVDModel(ABC):
    """Abstract base class for OVD model wrappers.

    All OVD model implementations should inherit from this class
    and implement the required methods.
    """

    def __init__(self, config: OVDConfig = None):
        """Initialize the OVD model.

        Args:
            config: Configuration for the model. Uses defaults if None.
        """
        self.config = config or OVDConfig()
        self._classes: List[str] = []
        self._model = None

    @abstractmethod
    def predict(self, image_path: str) -> DetectionResult:
        """Run inference on a single image.

        Args:
            image_path: Path to the input image

        Returns:
            DetectionResult containing detected boxes and metadata
        """
        raise NotImplementedError("Subclass must implement predict()")

    @abstractmethod
    def set_classes(self, classes: List[str]) -> None:
        """Set the text prompts for open-vocabulary detection.

        Args:
            classes: List of text prompts (one per class to detect)
        """
        raise NotImplementedError("Subclass must implement set_classes()")

    def get_classes(self) -> List[str]:
        """Get the currently set classes.

        Returns:
            List of class prompts
        """
        return self._classes.copy()

    def is_loaded(self) -> bool:
        """Check if the model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._model is not None

    def _load_model(self):
        """Load the model. Implemented by subclasses."""
        raise NotImplementedError("Subclass must implement _load_model()")
