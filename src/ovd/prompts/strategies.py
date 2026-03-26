"""Prompt strategies for OVD models.

This module defines different strategies for generating prompts
that guide OVD models to detect wooden building damage.
"""

from dataclasses import dataclass
from typing import List, Protocol


# Wooden building damage class definitions
DAMAGE_CLASSES = {
    "crack": "裂缝",
    "decay": "腐朽",
    "insect": "虫害",
    "mechanical": "机械损伤",
    "mildew": "霉变",
    "knot": "木节"
}


class PromptStrategy(Protocol):
    """Protocol for prompt strategies."""

    def generate(self) -> List[str]:
        """Generate a list of text prompts for OVD detection.

        Returns:
            List of text prompts, one per damage class.
        """
        ...

    def name(self) -> str:
        """Return the strategy name."""
        ...


@dataclass(frozen=True)
class BasicStrategy:
    """Basic strategy using simple class names.

    Example: ["crack", "decay", "insect", ...]
    """

    def name(self) -> str:
        return "basic"

    def generate(self) -> List[str]:
        return ["crack", "decay", "insect", "mechanical", "mildew", "knot"]


@dataclass(frozen=True)
class DetailedStrategy:
    """Detailed strategy using descriptive phrases.

    Example: ["wood crack or fracture on timber surface", ...]
    """

    def name(self) -> str:
        return "detailed"

    def generate(self) -> List[str]:
        return [
            "wood crack or fracture on timber surface",
            "decayed or rotten wood area",
            "insect holes or pest damage on wood",
            "mechanical damage or tool marks on wood",
            "mold or fungal growth on wood surface",
            "wood knot or natural tree node",
        ]


@dataclass(frozen=True)
class ContextStrategy:
    """Context-aware strategy with building context.

    Example: ["traditional wooden building damage: crack on timber beam", ...]
    """

    def name(self) -> str:
        return "context"

    def generate(self) -> List[str]:
        return [
            "traditional wooden building damage: crack on timber beam",
            "traditional wooden building damage: wood decay on pillar",
            "traditional wooden building damage: insect holes on wood",
            "traditional wooden building damage: mechanical damage",
            "traditional wooden building damage: mold or mildew",
            "traditional wooden building damage: wood knot",
        ]


@dataclass(frozen=True)
class ExpertStrategy:
    """Expert strategy optimized for CLIP text encoder.

    This strategy uses enhanced visual feature descriptions (color, texture, shape)
    that are optimized for CLIP's visual encoding capabilities.

    Example: "deep dark longitudinal structural crack on brown timber surface"
    """

    def name(self) -> str:
        return "expert"

    def generate(self) -> List[str]:
        return [
            "deep dark longitudinal structural crack on brown timber surface",
            "spongy dark brown decayed rotten wood texture",
            "cluster of tiny round black insect holes on wood",
            "sharp cut, gouge marks, or deep scratches on timber",
            "white or greenish fuzzy mold and mildew spots on wood",
            "dark circular natural wood knot with concentric rings"
        ]


def get_strategy(name: str) -> PromptStrategy:
    """Get a strategy by name.

    Args:
        name: Strategy name ('basic', 'detailed', 'context', or 'expert')

    Returns:
        PromptStrategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    strategies = {
        "basic": BasicStrategy(),
        "detailed": DetailedStrategy(),
        "context": ContextStrategy(),
        "expert": ExpertStrategy(),
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}")

    return strategies[name]


def list_strategies() -> List[str]:
    """List all available strategy names.

    Returns:
        List of strategy names
    """
    return ["basic", "detailed", "context", "expert"]
