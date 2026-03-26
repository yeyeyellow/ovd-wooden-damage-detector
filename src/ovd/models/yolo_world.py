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

        # Run inference with OOM prevention parameters
        start_time = time.time()

        # 【防爆显存】：强制注入 half=True 和 imgsz=640 防止 8GB 显卡 OOM
        results = self._model.predict(
            source=image_path,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            half=True,      # FP16 推理，节省显存
            imgsz=640,      # 限制输入尺寸，防止 OOM
            verbose=False,  # 安静模式
        )

        inference_time = time.time() - start_time

        # Parse results using native normalized coordinates
        boxes = []
        result = results[0]

        if hasattr(result, 'boxes') and result.boxes is not None:
            for i in range(len(result.boxes)):
                box_obj = result.boxes[i]
                # 【性能优化】：直接提取 Ultralytics 原生的高效归一化坐标 xywhn
                # 避免二次转换，直接使用原生的 normalized xywh 格式
                x_c, y_c, w, h = box_obj.xywhn[0].tolist()
                conf = float(box_obj.conf[0])
                cls_id = int(box_obj.cls[0])
                class_name = self._get_class_name(cls_id, result)

                # 将原生归一化坐标直接塞入对象
                # 注意：这里 x1=x_center, y1=y_center, x2=width, y2=height
                # 因为坐标已经是归一化的，上层可以直接使用
                boxes.append(BoundingBox(
                    x1=x_c,      # 归一化的中心 x
                    y1=y_c,      # 归一化的中心 y
                    x2=w,        # 归一化的宽度
                    y2=h,        # 归一化的高度
                    class_id=cls_id,
                    confidence=conf,
                    class_name=class_name
                ))

        return DetectionResult(
            image_path=image_path,
            boxes=boxes,
            inference_time=inference_time,
        )

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
