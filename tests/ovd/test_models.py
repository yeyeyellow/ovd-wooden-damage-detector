"""Tests for OVD model wrappers.

NOTE: These tests do not require actual models or GPU.
They test the interface and configuration logic only.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict

from scripts.ovd.models.base_ovd import (
    OVDConfig,
    DetectionResult,
    BoundingBox,
    BaseOVDModel,
)
from scripts.ovd.models.yolo_world import YOLOEModel


class TestOVDConfig:
    """Test OVDConfig dataclass."""

    def test_default_values(self):
        """OVDConfig should have sensible defaults."""
        config = OVDConfig()

        assert config.model_name == "yoloe-26x-seg.pt"
        assert config.conf_threshold == 0.25
        assert config.iou_threshold == 0.45
        assert config.max_detections == 300
        assert config.device == "cpu"

    def test_custom_values(self):
        """OVDConfig should accept custom values."""
        config = OVDConfig(
            model_name="yolov8x-worldv2.pt",
            conf_threshold=0.3,
            iou_threshold=0.5,
            device="cuda:0",
        )

        assert config.model_name == "yolov8x-worldv2.pt"
        assert config.conf_threshold == 0.3
        assert config.iou_threshold == 0.5
        assert config.device == "cuda:0"

    def test_is_immutable(self):
        """OVDConfig should be frozen (immutable)."""
        config = OVDConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.conf_threshold = 0.5


class TestBoundingBox:
    """Test BoundingBox dataclass."""

    def test_create_bbox(self):
        """Should create a bounding box with all fields."""
        bbox = BoundingBox(
            x1=100, y1=200, x2=300, y2=400,
            class_id=1, confidence=0.85, class_name="decay"
        )

        assert bbox.x1 == 100
        assert bbox.y1 == 200
        assert bbox.x2 == 300
        assert bbox.y2 == 400
        assert bbox.class_id == 1
        assert bbox.confidence == 0.85
        assert bbox.class_name == "decay"

    def test_bbox_width_and_height(self):
        """Should calculate width and height correctly."""
        bbox = BoundingBox(
            x1=100, y1=200, x2=300, y2=400,
            class_id=0, confidence=0.9, class_name="crack"
        )

        assert bbox.width == 200  # x2 - x1
        assert bbox.height == 200  # y2 - y1


class TestDetectionResult:
    """Test DetectionResult dataclass."""

    def test_create_result(self):
        """Should create a detection result."""
        result = DetectionResult(
            image_path="test.jpg",
            boxes=[],
            inference_time=0.15,
        )

        assert result.image_path == "test.jpg"
        assert result.boxes == []
        assert result.inference_time == 0.15

    def test_result_with_boxes(self):
        """Should create result with bounding boxes."""
        boxes = [
            BoundingBox(10, 20, 50, 60, 0, 0.9, "crack"),
            BoundingBox(100, 150, 200, 250, 1, 0.8, "decay"),
        ]
        result = DetectionResult(
            image_path="test.jpg",
            boxes=boxes,
            inference_time=0.2,
        )

        assert len(result.boxes) == 2
        assert result.boxes[0].class_name == "crack"
        assert result.boxes[1].class_name == "decay"


class TestYOLOEModel:
    """Test YOLOEModel wrapper."""

    def test_init_with_default_config(self):
        """Should initialize with default config."""
        model = YOLOEModel()

        assert model.config.model_name == "yoloe-26x-seg.pt"
        assert model.config.device == "cpu"

    def test_init_with_custom_config(self):
        """Should initialize with custom config."""
        config = OVDConfig(
            model_name="yolov8x-worldv2.pt",
            device="cuda:0",
        )
        model = YOLOEModel(config=config)

        assert model.config.model_name == "yolov8x-worldv2.pt"
        assert model.config.device == "cuda:0"

    def test_set_classes(self):
        """Should store custom classes."""
        model = YOLOEModel()
        classes = ["crack", "decay", "insect"]

        model.set_classes(classes)

        assert model._classes == classes

    def test_load_model_deferred(self):
        """Model should be loaded lazily on first predict."""
        from unittest.mock import patch
        import torch

        # Mock the ultralytics.YOLOE class (imported inside _load_model)
        with patch("ultralytics.YOLOE") as mock_yoloe:
            mock_model = MagicMock()
            mock_yoloe.return_value = mock_model

            # Mock predict to return empty results
            mock_result = MagicMock()
            mock_result.boxes = MagicMock()
            mock_result.boxes.data = torch.tensor([])
            mock_model.predict.return_value = [mock_result]

            model = YOLOEModel()

            # Model not loaded initially
            assert not model.is_loaded()

            # First predict should load the model
            result = model.predict("test.jpg")

            # Verify YOLOE was called to load model
            mock_yoloe.assert_called_once()

            # Model should now be loaded
            assert model.is_loaded()

    def test_predict_with_mocked_model(self):
        """Should call model.predict with correct parameters."""
        from unittest.mock import patch

        # Mock ultralytics.YOLOE class (imported inside _load_model)
        with patch("ultralytics.YOLOE") as mock_yoloe:
            import torch

            mock_model = MagicMock()
            mock_result = MagicMock()
            mock_result.boxes = MagicMock()
            mock_result.boxes.data = torch.tensor([
                [10, 20, 50, 60, 0.9, 0],  # x1, y1, x2, y2, conf, cls
            ])
            mock_result.names = {0: "crack"}
            mock_model.predict.return_value = [mock_result]
            mock_yoloe.return_value = mock_model

            model = YOLOEModel()
            model.set_classes(["crack"])

            result = model.predict("test.jpg")

            # Verify predict was called
            mock_model.predict.assert_called()

            # Verify result structure
            assert result.image_path == "test.jpg"
            assert len(result.boxes) == 1
            assert result.boxes[0].class_name == "crack"
            assert abs(result.boxes[0].confidence - 0.9) < 0.01  # Float tolerance

    def test_set_classes_called_on_model(self):
        """Should call set_classes on the underlying model."""
        mock_model = MagicMock()

        model = YOLOEModel()
        model._model = mock_model
        classes = ["crack", "decay", "insect"]

        model.set_classes(classes)

        # Verify stored in model
        assert model.get_classes() == classes

        # Verify set_classes was called on underlying model
        mock_model.set_classes.assert_called_once_with(classes)


class TestModelInterface:
    """Test BaseOVDModel interface."""

    def test_cannot_instantiate_base_class(self):
        """BaseOVDModel with abstract methods should raise TypeError."""
        # BaseOVDModel has abstract methods, so it can't be instantiated
        with pytest.raises(TypeError):
            BaseOVDModel()

    def test_complete_subclass_can_be_instantiated(self):
        """Complete subclass can be instantiated."""
        from scripts.ovd.models.base_ovd import DetectionResult

        # Create a minimal complete subclass
        class CompleteModel(BaseOVDModel):
            def predict(self, image_path: str) -> DetectionResult:
                return DetectionResult(image_path, [], 0.0)

            def set_classes(self, classes) -> None:
                self._classes = list(classes)

        # Should be able to instantiate
        model = CompleteModel()
        assert model is not None

        # Methods should work
        result = model.predict("test.jpg")
        assert result.image_path == "test.jpg"

        model.set_classes(["crack", "decay"])
        assert model.get_classes() == ["crack", "decay"]
