"""OVD (Open-Vocabulary Detection) Module

This module provides tools for exploring Open-Vocabulary Detection
techniques in the context of wooden building damage detection.
"""

from .prompts.builder import PromptBuilder
from .prompts.strategies import PromptStrategy, get_strategy
from .models import YOLOEModel, OVDConfig, DetectionResult, BoundingBox
from .label_generator import PseudoLabelGenerator, LabelGenerationConfig

__all__ = [
    "PromptBuilder",
    "PromptStrategy",
    "get_strategy",
    "YOLOEModel",
    "OVDConfig",
    "DetectionResult",
    "BoundingBox",
    "PseudoLabelGenerator",
    "LabelGenerationConfig",
]
