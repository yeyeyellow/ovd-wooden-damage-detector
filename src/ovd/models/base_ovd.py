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

    Attributes:
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates
        class_id: Class index (0-based)
        confidence: Detection confidence score (0-1)
        class_name: Human-readable class name
    """

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float
    class_name: str

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> tuple:
        """Center point (x, y) of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class DetectionResult:
    """Result of OVD inference on a single image.

    Attributes:
        image_path: Path to the input image
        boxes: List of detected bounding boxes
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
