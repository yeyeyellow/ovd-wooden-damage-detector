"""Tests for YOLO format utilities and pseudo-label generator."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from scripts.ovd.label_generator import (
    YOLOLabel,
    LabelGenerationConfig,
    PseudoLabelGenerator,
    bbox_to_yolo,
    get_image_size,
)


class TestYOLOLabel:
    """Test YOLOLabel dataclass."""

    def test_create_yolo_label(self):
        """Should create a YOLO label with all fields."""
        label = YOLOLabel(
            class_id=0,
            x_center=0.5,
            y_center=0.5,
            width=0.25,
            height=0.3
        )

        assert label.class_id == 0
        assert label.x_center == 0.5
        assert label.y_center == 0.5
        assert label.width == 0.25
        assert label.height == 0.3

    def test_to_yolo_string(self):
        """Should convert to YOLO format string."""
        label = YOLOLabel(
            class_id=2,
            x_center=0.123456,
            y_center=0.789012,
            width=0.25,
            height=0.3
        )

        result = label.to_yolo_string()

        assert result == "2 0.123456 0.789012 0.250000 0.300000"

    def test_to_yolo_string_rounding(self):
        """Should round to 6 decimal places."""
        label = YOLOLabel(
            class_id=0,
            x_center=0.123456789,
            y_center=0.987654321,
            width=0.111111111,
            height=0.999999999
        )

        result = label.to_yolo_string()

        assert result == "0 0.123457 0.987654 0.111111 1.000000"


class TestBBoxToYOLO:
    """Test bbox_to_yolo conversion function."""

    def test_convert_bbox_to_yolo_format(self):
        """Should convert bbox coordinates to YOLO format."""
        x_center, y_center, width, height = bbox_to_yolo(
            x1=100, y1=100, x2=300, y2=400,
            img_width=640, img_height=480
        )

        # Expected: center at (200, 250), size (200, 300)
        # Normalized: (200/640, 250/480, 200/640, 300/480)
        assert abs(x_center - 0.3125) < 0.001
        assert abs(y_center - 0.5208) < 0.001
        assert abs(width - 0.3125) < 0.001
        assert abs(height - 0.625) < 0.001

    def test_full_image_bbox(self):
        """Should handle bbox covering entire image."""
        x_center, y_center, width, height = bbox_to_yolo(
            x1=0, y1=0, x2=640, y2=480,
            img_width=640, img_height=480
        )

        assert x_center == 0.5
        assert y_center == 0.5
        assert width == 1.0
        assert height == 1.0

    def test_small_bbox(self):
        """Should handle small bbox."""
        x_center, y_center, width, height = bbox_to_yolo(
            x1=320, y1=240, x2=330, y2=250,
            img_width=640, img_height=480
        )

        assert abs(x_center - 0.5078) < 0.001
        assert abs(y_center - 0.5104) < 0.001
        assert abs(width - 0.0156) < 0.001
        assert abs(height - 0.0208) < 0.001


class TestGetImageSize:
    """Test get_image_size function."""

    def test_get_jpg_image_size(self, tmp_path):
        """Should get size of JPEG image."""
        # Create a test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (640, 480), color="red")
        img.save(img_path)

        width, height = get_image_size(str(img_path))

        assert width == 640
        assert height == 480

    def test_get_png_image_size(self, tmp_path):
        """Should get size of PNG image."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (800, 600), color="blue")
        img.save(img_path)

        width, height = get_image_size(str(img_path))

        assert width == 800
        assert height == 600

    def test_nonexistent_file_raises_error(self):
        """Should raise error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            get_image_size("nonexistent.jpg")


class TestLabelGenerationConfig:
    """Test LabelGenerationConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = LabelGenerationConfig(
            images_dir="data/images",
            output_dir="data/labels"
        )

        assert config.conf_threshold == 0.25
        assert config.image_width == 640
        assert config.image_height == 640
        assert config.create_output_dir is True

    def test_custom_values(self):
        """Should accept custom values."""
        config = LabelGenerationConfig(
            images_dir="data/images",
            output_dir="data/labels",
            conf_threshold=0.5,
            image_width=1280,
            image_height=720,
            create_output_dir=False
        )

        assert config.conf_threshold == 0.5
        assert config.image_width == 1280
        assert config.image_height == 720
        assert config.create_output_dir is False


class TestPseudoLabelGenerator:
    """Test PseudoLabelGenerator class."""

    def test_init_with_config(self):
        """Should initialize with model and config."""
        mock_model = MagicMock()
        config = LabelGenerationConfig(
            images_dir="data/images",
            output_dir="data/labels"
        )

        generator = PseudoLabelGenerator(mock_model, config)

        assert generator.model == mock_model
        assert generator.config == config

    def test_init_with_default_config(self):
        """Should use default config if none provided."""
        mock_model = MagicMock()

        generator = PseudoLabelGenerator(mock_model)

        assert generator.model == mock_model
        assert generator.config is not None

    def test_generate_single_returns_yolo_labels(self, tmp_path):
        """Should generate YOLO labels for a single image."""
        # Create test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (640, 480), color="red")
        img.save(img_path)

        # Mock model prediction
        mock_model = MagicMock()
        from scripts.ovd.models.base_ovd import DetectionResult, BoundingBox
        mock_model.predict.return_value = DetectionResult(
            image_path=str(img_path),
            boxes=[
                BoundingBox(100, 100, 300, 400, 0, 0.85, "crack"),
                BoundingBox(400, 200, 600, 350, 1, 0.75, "decay"),
            ],
            inference_time=0.15
        )

        config = LabelGenerationConfig(
            images_dir=str(tmp_path),
            output_dir=str(tmp_path / "labels")
        )
        generator = PseudoLabelGenerator(mock_model, config)

        labels = generator.generate_single(str(img_path))

        assert len(labels) == 2
        assert labels[0].class_id == 0
        assert labels[1].class_id == 1

    def test_generate_single_filters_by_confidence(self, tmp_path):
        """Should filter detections by confidence threshold."""
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (640, 480))
        img.save(img_path)

        mock_model = MagicMock()
        from scripts.ovd.models.base_ovd import DetectionResult, BoundingBox
        mock_model.predict.return_value = DetectionResult(
            image_path=str(img_path),
            boxes=[
                BoundingBox(100, 100, 300, 400, 0, 0.85, "crack"),   # Above threshold
                BoundingBox(400, 200, 600, 350, 1, 0.15, "decay"),   # Below threshold
            ],
            inference_time=0.15
        )

        config = LabelGenerationConfig(
            images_dir=str(tmp_path),
            output_dir=str(tmp_path / "labels"),
            conf_threshold=0.25
        )
        generator = PseudoLabelGenerator(mock_model, config)

        labels = generator.generate_single(str(img_path))

        # Only crack should be kept (conf > 0.25)
        assert len(labels) == 1
        assert labels[0].class_id == 0

    def test_save_labels_creates_file(self, tmp_path):
        """Should save labels to file."""
        mock_model = MagicMock()
        config = LabelGenerationConfig(
            images_dir="data/images",
            output_dir=str(tmp_path / "labels")
        )
        generator = PseudoLabelGenerator(mock_model, config)

        labels = [
            YOLOLabel(0, 0.5, 0.5, 0.25, 0.3),
            YOLOLabel(1, 0.7, 0.7, 0.15, 0.2),
        ]

        output_path = generator.save_labels("data/images/test.jpg", labels)

        assert output_path.exists()
        assert output_path.name == "test.txt"

        # Verify content
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "0 0.500000 0.500000 0.250000 0.300000"
        assert lines[1] == "1 0.700000 0.700000 0.150000 0.200000"

    def test_save_labels_creates_output_dir(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        mock_model = MagicMock()
        config = LabelGenerationConfig(
            images_dir="data/images",
            output_dir=str(tmp_path / "new_dir" / "labels"),
            create_output_dir=True
        )
        generator = PseudoLabelGenerator(mock_model, config)

        labels = [YOLOLabel(0, 0.5, 0.5, 0.25, 0.3)]

        output_path = generator.save_labels("data/images/test.jpg", labels)

        assert output_path.parent.exists()
        assert output_path.exists()

    def test_generate_batch_returns_stats(self, tmp_path):
        """Should return generation statistics."""
        # Create test images
        for i in range(3):
            img_path = tmp_path / f"test{i}.jpg"
            img = Image.new("RGB", (640, 480))
            img.save(img_path)

        mock_model = MagicMock()
        from scripts.ovd.models.base_ovd import DetectionResult, BoundingBox
        mock_model.predict.return_value = DetectionResult(
            image_path="dummy",
            boxes=[BoundingBox(100, 100, 300, 400, 0, 0.85, "crack")],
            inference_time=0.15
        )

        config = LabelGenerationConfig(
            images_dir=str(tmp_path),
            output_dir=str(tmp_path / "labels")
        )
        generator = PseudoLabelGenerator(mock_model, config)

        stats = generator.generate_batch()

        assert stats["total"] == 3
        assert stats["successful"] == 3
        assert stats["failed"] == 0
        assert stats["total_detections"] == 3

    def test_generate_batch_handles_failures(self, tmp_path):
        """Should handle individual failures gracefully."""
        # Create test images
        for i in range(3):
            img_path = tmp_path / f"test{i}.jpg"
            img = Image.new("RGB", (640, 480))
            img.save(img_path)

        mock_model = MagicMock()
        from scripts.ovd.models.base_ovd import DetectionResult, BoundingBox

        # Make second image fail
        call_count = [0]
        def side_effect(path):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Detection failed")
            return DetectionResult(
                image_path=path,
                boxes=[BoundingBox(100, 100, 300, 400, 0, 0.85, "crack")],
                inference_time=0.15
            )

        mock_model.predict.side_effect = side_effect

        config = LabelGenerationConfig(
            images_dir=str(tmp_path),
            output_dir=str(tmp_path / "labels")
        )
        generator = PseudoLabelGenerator(mock_model, config)

        stats = generator.generate_batch()

        assert stats["total"] == 3
        assert stats["successful"] == 2
        assert stats["failed"] == 1
