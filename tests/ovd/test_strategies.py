"""Tests for prompt strategies."""

import pytest

from scripts.ovd.prompts.strategies import (
    PromptStrategy,
    BasicStrategy,
    DetailedStrategy,
    ContextStrategy,
    get_strategy,
    list_strategies,
    DAMAGE_CLASSES,
)


class TestBasicStrategy:
    """Test BasicStrategy prompt generation."""

    def test_generate_returns_six_prompts(self):
        """Basic strategy should return exactly 6 prompts."""
        strategy = BasicStrategy()
        prompts = strategy.generate()

        assert len(prompts) == 6

    def test_generate_returns_simple_class_names(self):
        """Basic strategy should return simple English class names."""
        strategy = BasicStrategy()
        prompts = strategy.generate()

        expected = ["crack", "decay", "insect", "mechanical", "mildew", "knot"]
        assert prompts == expected

    def test_name_returns_basic(self):
        """Basic strategy name should be 'basic'."""
        strategy = BasicStrategy()
        assert strategy.name() == "basic"


class TestDetailedStrategy:
    """Test DetailedStrategy prompt generation."""

    def test_generate_returns_six_prompts(self):
        """Detailed strategy should return exactly 6 prompts."""
        strategy = DetailedStrategy()
        prompts = strategy.generate()

        assert len(prompts) == 6

    def test_generate_returns_descriptive_phrases(self):
        """Detailed strategy should return descriptive phrases."""
        strategy = DetailedStrategy()
        prompts = strategy.generate()

        # Each prompt should be a descriptive phrase
        for prompt in prompts:
            assert " " in prompt, f"Prompt '{prompt}' should contain spaces"
            assert len(prompt) > 10, f"Prompt '{prompt}' should be descriptive"

    def test_contains_crack_description(self):
        """Detailed strategy should describe crack properly."""
        strategy = DetailedStrategy()
        prompts = strategy.generate()

        crack_prompt = prompts[0]
        assert "crack" in crack_prompt.lower()
        assert "wood" in crack_prompt.lower()

    def test_name_returns_detailed(self):
        """Detailed strategy name should be 'detailed'."""
        strategy = DetailedStrategy()
        assert strategy.name() == "detailed"


class TestContextStrategy:
    """Test ContextStrategy prompt generation."""

    def test_generate_returns_six_prompts(self):
        """Context strategy should return exactly 6 prompts."""
        strategy = ContextStrategy()
        prompts = strategy.generate()

        assert len(prompts) == 6

    def test_generate_includes_building_context(self):
        """Context strategy should include traditional building context."""
        strategy = ContextStrategy()
        prompts = strategy.generate()

        for prompt in prompts:
            assert "traditional" in prompt.lower() or "wooden" in prompt.lower()

    def test_name_returns_context(self):
        """Context strategy name should be 'context'."""
        strategy = ContextStrategy()
        assert strategy.name() == "context"


class TestGetStrategy:
    """Test get_strategy factory function."""

    def test_get_basic_strategy(self):
        """Should return BasicStrategy for 'basic'."""
        strategy = get_strategy("basic")
        assert isinstance(strategy, BasicStrategy)

    def test_get_detailed_strategy(self):
        """Should return DetailedStrategy for 'detailed'."""
        strategy = get_strategy("detailed")
        assert isinstance(strategy, DetailedStrategy)

    def test_get_context_strategy(self):
        """Should return ContextStrategy for 'context'."""
        strategy = get_strategy("context")
        assert isinstance(strategy, ContextStrategy)

    def test_invalid_name_raises_value_error(self):
        """Should raise ValueError for unknown strategy name."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("unknown")


class TestListStrategies:
    """Test list_strategies function."""

    def test_returns_all_strategy_names(self):
        """Should return list of all available strategy names."""
        strategies = list_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) >= 3
        assert "basic" in strategies
        assert "detailed" in strategies
        assert "context" in strategies


class TestDamageClasses:
    """Test DAMAGE_CLASSES constant."""

    def test_contains_all_six_classes(self):
        """DAMAGE_CLASSES should contain exactly 6 entries."""
        assert len(DAMAGE_CLASSES) == 6

    def test_contains_expected_keys(self):
        """DAMAGE_CLASSES should have expected English keys."""
        expected_keys = {"crack", "decay", "insect", "mechanical", "mildew", "knot"}
        assert set(DAMAGE_CLASSES.keys()) == expected_keys

    def test_values_are_chinese(self):
        """DAMAGE_CLASSES values should be Chinese names."""
        expected_values = {"裂缝", "腐朽", "虫害", "机械损伤", "霉变", "木节"}
        assert set(DAMAGE_CLASSES.values()) == expected_values
