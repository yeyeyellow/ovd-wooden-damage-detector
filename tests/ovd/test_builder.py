"""Tests for PromptBuilder."""

import pytest

from scripts.ovd.prompts.builder import PromptBuilder, PromptConfig


class TestPromptConfig:
    """Test PromptConfig dataclass."""

    def test_default_values(self):
        """PromptConfig should have sensible defaults."""
        config = PromptConfig()

        assert config.strategy_name == "basic"
        assert config.custom_prompts is None
        assert config.language == "en"

    def test_custom_values(self):
        """PromptConfig should accept custom values."""
        config = PromptConfig(
            strategy_name="detailed",
            custom_prompts=["custom prompt"],
            language="zh"
        )

        assert config.strategy_name == "detailed"
        assert config.custom_prompts == ["custom prompt"]
        assert config.language == "zh"

    def test_is_immutable(self):
        """PromptConfig should be frozen (immutable)."""
        config = PromptConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.strategy_name = "detailed"


class TestPromptBuilder:
    """Test PromptBuilder class."""

    def test_build_returns_prompts(self):
        """build() should return a list of prompts."""
        builder = PromptBuilder()
        prompts = builder.build()

        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_build_returns_six_prompts_by_default(self):
        """build() should return 6 prompts by default (one per class)."""
        builder = PromptBuilder()
        prompts = builder.build()

        assert len(prompts) == 6

    def test_with_strategy_returns_new_builder(self):
        """with_strategy() should return a new builder instance."""
        builder = PromptBuilder()
        new_builder = builder.with_strategy("detailed")

        assert new_builder is not builder
        assert isinstance(new_builder, PromptBuilder)

    def test_with_strategy_modifies_strategy(self):
        """with_strategy() should change the strategy."""
        builder = PromptBuilder()
        new_builder = builder.with_strategy("detailed")

        assert new_builder.config.strategy_name == "detailed"
        # Original should be unchanged
        assert builder.config.strategy_name == "basic"

    def test_with_custom_prompts_returns_new_builder(self):
        """with_custom_prompts() should return a new builder instance."""
        builder = PromptBuilder()
        new_builder = builder.with_custom_prompts(["custom", "prompts"])

        assert new_builder is not builder
        assert isinstance(new_builder, PromptBuilder)

    def test_with_custom_prompts_uses_custom(self):
        """with_custom_prompts() should override strategy prompts."""
        builder = PromptBuilder()
        new_builder = builder.with_custom_prompts(["custom", "prompts"])
        prompts = new_builder.build()

        assert prompts == ["custom", "prompts"]

    def test_clear_cache(self):
        """clear_cache() should reset the internal cache."""
        builder = PromptBuilder()
        # Build to create cache
        _ = builder.build()

        builder.clear_cache()
        # Cache should be cleared (no public API to verify,
        # but this should not raise an error)

    def test_chaining(self):
        """Builder methods should be chainable."""
        builder = PromptBuilder()
        prompts = (builder
                   .with_strategy("detailed")
                   .with_custom_prompts(["test"])
                   .build())

        assert prompts == ["test"]
