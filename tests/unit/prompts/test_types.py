"""Tests for arize.prompts.types public re-exports."""

from __future__ import annotations

import pytest

import arize.prompts.types as types_module
from arize.prompts.types import (
    InputVariableFormat,
    InvocationParams,
    LLMMessage,
    LlmProvider,
    Prompt,
    PromptsList200Response,
    PromptVersion,
    PromptVersionLabelsSet200Response,
    PromptVersionsList200Response,
    PromptWithVersion,
    ProviderParams,
)


@pytest.mark.unit
class TestPromptsTypes:
    """Tests for the prompts types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        expected = {
            "InputVariableFormat",
            "InvocationParams",
            "LLMMessage",
            "LlmProvider",
            "Prompt",
            "PromptVersion",
            "PromptVersionLabelsSet200Response",
            "PromptVersionsList200Response",
            "PromptWithVersion",
            "PromptsList200Response",
            "ProviderParams",
        }
        assert expected.issubset(set(types_module.__all__))

    def test_input_variable_format_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(InputVariableFormat, Enum)

    def test_llm_provider_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(LlmProvider, Enum)

    @pytest.mark.parametrize(
        "cls",
        [
            InvocationParams,
            LLMMessage,
            Prompt,
            PromptVersion,
            PromptVersionLabelsSet200Response,
            PromptVersionsList200Response,
            PromptWithVersion,
            PromptsList200Response,
            ProviderParams,
        ],
    )
    def test_type_is_class(self, cls: type) -> None:
        assert isinstance(cls, type)
