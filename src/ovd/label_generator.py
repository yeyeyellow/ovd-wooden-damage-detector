"""YOLO format utilities for OVD pseudo-label generation.

This module provides utilities to convert OVD detection results
to YOLO format and save them as label files.
"""

from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass(frozen=True)
class YOLOLabel:
    """A single YOLO format label.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values are normalized to [0, 1].

    Attributes:
        class_id: Class index (0-based)
        x_center: Center x coordinate (normalized)
        y_center: Center y coordinate (normalized)
        width: Box width (normalized)
        height: Box height (normalized)
    """

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_string(self) -> str:
        """Convert to YOLO format string.

        Returns:
            String in format: "class_id x_center y_center width height"
        """
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


@dataclass
class LabelGenerationConfig:
    """Configuration for pseudo-label generation.

    Attributes:
        images_dir: Directory containing input images
        output_dir: Directory to save YOLO label files
        conf_threshold: Minimum confidence to keep a detection
        create_output_dir: Whether to create output directory if missing
    """

    images_dir: str
    output_dir: str
    conf_threshold: float = 0.25
    create_output_dir: bool = True


class PseudoLabelGenerator:
    """Generate pseudo-labels using OVD model.

    This class orchestrates the process of:
    1. Running OVD detection on images
    2. Converting results to YOLO format
    3. Saving label files

    【优化】直接使用底层传递的归一化坐标，无需二次转换。
    """

    def __init__(self, model, config: LabelGenerationConfig = None):
        """Initialize the pseudo-label generator.

        Args:
            model: OVD model instance (e.g., YOLOEModel)
            config: Generation configuration
        """
        self.model = model
        self.config = config or LabelGenerationConfig(
            images_dir="data/images",
            output_dir="data/labels"
        )

        # Create output directory if needed
        if self.config.create_output_dir:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def generate_single(self, image_path: str) -> List[YOLOLabel]:
        """Generate YOLO labels for a single image.

        【优化】直接使用底层传递上来的原生归一化数据，
        无需调用 PIL 读取图片尺寸，也无需重复计算归一化坐标。

        Args:
            image_path: Path to input image

        Returns:
            List of YOLOLabel objects
        """
        # Run OVD detection
        result = self.model.predict(image_path)

        # 【性能优化】直接使用底层传递的归一化坐标
        # 底层 YOLOE 现在直接返回: x1=x_center, y1=y_center, x2=width, y2=height
        # 这些值已经是归一化的 [0, 1]，可以直接使用
        labels = []
        for box in result.boxes:
            # Filter by confidence
            if box.confidence < self.config.conf_threshold:
                continue

            # 直接使用底层传递的原生归一化数据
            labels.append(YOLOLabel(
                class_id=box.class_id,
                x_center=box.x1,     # 归一化的中心 x
                y_center=box.y1,     # 归一化的中心 y
                width=box.x2,        # 归一化的宽度
                height=box.y2        # 归一化的高度
            ))

        return labels

    def generate_batch(self, image_paths: List[str] = None) -> dict:
        """Generate YOLO labels for multiple images.

        Args:
            image_paths: List of image paths. If None, uses all images
                        from config.images_dir

        Returns:
            Dictionary with generation statistics:
            {
                "total": int,
                "successful": int,
                "failed": int,
                "total_detections": int,
                "results": List[dict]
            }
        """
        import glob

        # Get image paths if not provided
        if image_paths is None:
            image_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                image_paths.extend(glob.glob(str(Path(self.config.images_dir) / ext)))

        results = []
        successful = 0
        failed = 0
        total_detections = 0

        for img_path in image_paths:
            try:
                labels = self.generate_single(img_path)
                self.save_labels(img_path, labels)

                results.append({
                    "image": img_path,
                    "labels": len(labels),
                    "status": "success"
                })
                successful += 1
                total_detections += len(labels)

            except Exception as e:
                results.append({
                    "image": img_path,
                    "labels": 0,
                    "status": "failed",
                    "error": str(e)
                })
                failed += 1

        return {
            "total": len(image_paths),
            "successful": successful,
            "failed": failed,
            "total_detections": total_detections,
            "results": results
        }

    def save_labels(self, image_path: str, labels: List[YOLOLabel]) -> Path:
        """Save YOLO labels to file.

        Args:
            image_path: Original image path (used to derive label filename)
            labels: List of YOLOLabel objects

        Returns:
            Path to saved label file
        """
        # Derive label filename from image path
        image_name = Path(image_path).stem
        label_path = Path(self.config.output_dir) / f"{image_name}.txt"

        # Create output directory if needed
        label_path.parent.mkdir(parents=True, exist_ok=True)

        # Write labels
        with open(label_path, "w") as f:
            for label in labels:
                f.write(label.to_yolo_string() + "\n")

        return label_path
