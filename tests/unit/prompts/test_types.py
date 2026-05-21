"""Tests for arize.prompts.types public re-exports."""

from __future__ import annotations

from typing import Any

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


@pytest.mark.unit
class TestInvocationParamsForwardCompat:
    """Regression tests for forward-compatibility of InvocationParams.

    Model providers add new parameters frequently. These tests ensure that
    unknown fields returned by the API do not cause deserialization errors,
    and that such fields are preserved on round-trip (so callers can still
    read them via .additional_properties).

    Regression: https://github.com/Arize-ai/arize/issues/72582
    """

    def test_from_dict_tolerates_unknown_fields(self) -> None:
        """from_dict() must not raise when the API returns unrecognised fields."""
        payload: dict[str, Any] = {
            "temperature": 0.7,
            "future_param_from_new_model": "some_value",
        }
        # Must not raise ValueError
        result = InvocationParams.from_dict(payload)
        assert result is not None

    def test_unknown_fields_stored_in_additional_properties(self) -> None:
        """Unknown fields should be accessible via .additional_properties."""
        payload: dict[str, Any] = {
            "temperature": 0.5,
            "verbosity": "medium",
            "some_future_param": 42,
        }
        result = InvocationParams.from_dict(payload)
        assert result is not None
        assert result.temperature == 0.5
        assert result.verbosity == "medium"
        assert result.additional_properties["some_future_param"] == 42

    def test_unknown_fields_round_trip_through_to_dict(self) -> None:
        """to_dict() should include unknown fields so they survive a read-modify-write."""
        payload: dict[str, Any] = {
            "temperature": 0.9,
            "a_brand_new_provider_param": "xhigh",
        }
        result = InvocationParams.from_dict(payload)
        assert result is not None
        serialized = result.to_dict()
        assert serialized["temperature"] == 0.9
        assert serialized["a_brand_new_provider_param"] == "xhigh"

    def test_known_fields_still_typed_correctly(self) -> None:
        """Adding additional_properties should not break known-field construction."""
        params = InvocationParams(
            temperature=0.3, max_tokens=100, verbosity="low"
        )
        assert params.temperature == 0.3
        assert params.max_tokens == 100
        assert params.verbosity == "low"
        assert params.additional_properties == {}


@pytest.mark.unit
class TestProviderParamsForwardCompat:
    """Forward-compatibility tests for ProviderParams.

    Provider-specific parameters are similarly open-ended — cloud providers
    add options without notice. Unknown fields must not crash deserialization.
    """

    def test_from_dict_tolerates_unknown_fields(self) -> None:
        """from_dict() must not raise when the API returns unrecognised provider fields."""
        payload: dict[str, Any] = {
            "anthropic_version": "2023-06-01",
            "new_provider_option_we_dont_know_about": True,
        }
        result = ProviderParams.from_dict(payload)
        assert result is not None

    def test_unknown_fields_stored_in_additional_properties(self) -> None:
        """Unknown ProviderParams fields should land in .additional_properties."""
        payload: dict[str, Any] = {
            "region": "us-east-1",
            "experimental_turbo_mode": True,
        }
        result = ProviderParams.from_dict(payload)
        assert result is not None
        assert result.region == "us-east-1"
        assert result.additional_properties["experimental_turbo_mode"] is True
