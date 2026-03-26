"""YOLOE (YOLO-World) model wrapper.

This module provides a wrapper for the YOLOE/YOLO-World model
from Ultralytics for open-vocabulary object detection.
"""

import time
from typing import List

from .base_ovd import BaseOVDModel, OVDConfig, DetectionResult, BoundingBox


class YOLOEModel(BaseOVDModel):
    """Wrapper for YOLOE (YOLO-World) model.

    This model supports open-vocabulary detection through text prompts.
    Reference: https://github.com/AILab-CVC/YOLO-World

    Example usage:
        ```python
        model = YOLOEModel()
        model.set_classes(["crack", "decay", "insect"])
        result = model.predict("image.jpg")
        ```
    """

    def __init__(self, config: OVDConfig = None):
        """Initialize YOLOE model.

        Args:
            config: Model configuration
        """
        super().__init__(config)

    def _load_model(self):
        """Lazy-load the YOLOE model."""
        if self._model is None:
            try:
                from ultralytics import YOLOE
                self._model = YOLOE(self.config.model_name)
            except ImportError as e:
                raise ImportError(
                    "ultralytics with YOLOE support is required. "
                    "Install with: pip install ultralytics"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load YOLOE model '{self.config.model_name}': {e}"
                ) from e

    def set_classes(self, classes: List[str]) -> None:
        """Set the text prompts for open-vocabulary detection.

        This method updates both the stored classes and the underlying model.

        Args:
            classes: List of text prompts (one per class to detect)
        """
        self._classes = classes.copy()

        # If model is already loaded, update it
        if self._model is not None:
            self._model.set_classes(classes)

    def predict(self, image_path: str) -> DetectionResult:
        """Run inference on a single image.

        Args:
            image_path: Path to the input image

        Returns:
            DetectionResult containing detected boxes and metadata
        """
        # Lazy-load model if not already loaded
        if not self.is_loaded():
            self._load_model()

        # Set classes if not already set
        if self._classes and hasattr(self._model, 'set_classes'):
            # Only set if model doesn't have classes set
            # (check if model's internal classes differ)
            try:
                current = getattr(self._model, 'classes', None)
                if current != self._classes:
                    self._model.set_classes(self._classes)
            except Exception:
                pass  # Model may not support set_classes yet

        import torch

        # Run inference
        start_time = time.time()

        try:
            # Use segmentation predictor if available
            from ultralytics.models.yolo.yolo.yoloe import YOLOEVPSegPredictor

            results = self._model.predict(
                source=image_path,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                device=self.config.device,
                predictor=YOLOEVPSegPredictor,
                stream=False,
            )
        except Exception:
            # Fallback to standard predictor
            results = self._model.predict(
                source=image_path,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                device=self.config.device,
                stream=False,
            )

        inference_time = time.time() - start_time

        # Parse results
        boxes = self._parse_results(results)

        return DetectionResult(
            image_path=image_path,
            boxes=boxes,
            inference_time=inference_time,
        )

    def _parse_results(self, results) -> List[BoundingBox]:
        """Parse YOLOE results into BoundingBox objects.

        Args:
            results: Raw results from YOLOE model

        Returns:
            List of BoundingBox objects
        """
        import torch

        boxes = []

        for result in results:
            # result.boxes.data is a tensor with shape [N, 6]
            # columns: x1, y1, x2, y2, confidence, class_id
            if hasattr(result, 'boxes') and result.boxes is not None:
                box_data = result.boxes.data

                # Convert to numpy if tensor
                if isinstance(box_data, torch.Tensor):
                    box_data = box_data.cpu().numpy()

                for row in box_data:
                    x1, y1, x2, y2, conf, cls_id = row

                    # Get class name from model names if available
                    class_name = self._get_class_name(int(cls_id), result)

                    boxes.append(BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        class_id=int(cls_id),
                        confidence=float(conf),
                        class_name=class_name,
                    ))

        return boxes

    def _get_class_name(self, class_id: int, result) -> str:
        """Get the class name for a given class ID.

        Args:
            class_id: Class index
            result: YOLOE result object

        Returns:
            Class name as string
        """
        # Try to get from result.names
        if hasattr(result, 'names') and result.names:
            name = result.names.get(class_id)
            if name:
                return name

        # Fallback to stored classes
        if 0 <= class_id < len(self._classes):
            return self._classes[class_id]

        # Default fallback
        return f"class_{class_id}"

    def predict_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """Run inference on multiple images.

        Args:
            image_paths: List of paths to input images

        Returns:
            List of DetectionResult objects
        """
        return [self.predict(path) for path in image_paths]
