"""Prompt builder for OVD models.

This module provides a builder pattern for constructing
and managing OVD detection prompts.
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional


@dataclass(frozen=True)
class PromptConfig:
    """Configuration for prompt building.

    Attributes:
        strategy_name: Name of the prompt strategy to use
        custom_prompts: Optional custom prompts override
        language: Language for prompts ('en' or 'zh')
    """

    strategy_name: str = "basic"
    custom_prompts: Optional[List[str]] = None
    language: str = "en"


@dataclass
class PromptBuilder:
    """Builder for constructing OVD detection prompts.

    This class manages prompt generation with different strategies
    and allows customization.
    """

    config: PromptConfig = field(default_factory=PromptConfig)
    _cache: Optional[List[str]] = field(default=None, init=False, repr=False)

    def build(self) -> List[str]:
        """Build prompts based on current configuration.

        Returns:
            List of text prompts for OVD detection
        """
        if self._cache is not None:
            return self._cache

        if self.config.custom_prompts is not None:
            self._cache = self.config.custom_prompts
        else:
            from .strategies import get_strategy
            strategy = get_strategy(self.config.strategy_name)
            self._cache = strategy.generate()

        return self._cache

    def with_strategy(self, strategy_name: str) -> "PromptBuilder":
        """Return a new builder with the specified strategy.

        Args:
            strategy_name: Name of the strategy to use

        Returns:
            New PromptBuilder instance
        """
        new_config = replace(self.config, strategy_name=strategy_name)
        return PromptBuilder(config=new_config)

    def with_custom_prompts(self, prompts: List[str]) -> "PromptBuilder":
        """Return a new builder with custom prompts.

        Args:
            prompts: List of custom prompt strings

        Returns:
            New PromptBuilder instance
        """
        new_config = replace(self.config, custom_prompts=prompts)
        return PromptBuilder(config=new_config)

    def clear_cache(self) -> None:
        """Clear the cached prompts.

        Call this after changing configuration.
        """
        self._cache = None
