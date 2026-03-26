"""YOLO format utilities for OVD pseudo-label generation.

This module provides utilities to convert OVD detection results
to YOLO format and save them as label files.
"""

from dataclasses import dataclass
from typing import List, Tuple
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
        image_width: Reference image width for normalization
        image_height: Reference image height for normalization
        create_output_dir: Whether to create output directory if missing
    """

    images_dir: str
    output_dir: str
    conf_threshold: float = 0.25
    image_width: int = 640
    image_height: int = 640
    create_output_dir: bool = True


class PseudoLabelGenerator:
    """Generate pseudo-labels using OVD model.

    This class orchestrates the process of:
    1. Running OVD detection on images
    2. Converting results to YOLO format
    3. Saving label files
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

        Args:
            image_path: Path to input image

        Returns:
            List of YOLOLabel objects
        """
        from .models.base_ovd import BoundingBox

        # Get actual image size
        img_width, img_height = get_image_size(image_path)

        # Run OVD detection
        result = self.model.predict(image_path)

        # Convert to YOLO format
        labels = []
        for box in result.boxes:
            # Filter by confidence
            if box.confidence < self.config.conf_threshold:
                continue

            # Convert to YOLO format
            x_center, y_center, width, height = bbox_to_yolo(
                box.x1, box.y1, box.x2, box.y2,
                img_width, img_height
            )

            labels.append(YOLOLabel(
                class_id=box.class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height
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


def bbox_to_yolo(
    x1: float, y1: float, x2: float, y2: float,
    img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """Convert bounding box to YOLO format.

    Args:
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates
        img_width: Image width
        img_height: Image height

    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1]
    """
    # Calculate center and size
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2

    # Normalize to [0, 1]
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    # Clamp to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def get_image_size(image_path: str) -> Tuple[int, int]:
    """Get image dimensions.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (width, height)
    """
    from PIL import Image

    with Image.open(image_path) as img:
        return img.size
