"""Tests for arize.prompts.types public re-exports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

import arize.prompts.types as types_module
from arize._generated.api_client.models.input_variable_format import (
    InputVariableFormat,
)
from arize._generated.api_client.models.llm_provider import LlmProvider
from arize._generated.api_client.models.message_role import MessageRole
from arize._generated.api_client.models.tool_call import ToolCall
from arize._generated.api_client.models.tool_call_function import (
    ToolCallFunction,
)
from arize._generated.api_client.models.tool_call_type import ToolCallType
from arize.prompts.types import (
    InvocationParams,
    ListPromptsResponse,
    ListPromptVersionsResponse,
    LLMMessage,
    Prompt,
    PromptVersion,
    PromptWithVersion,
    ProviderParams,
)

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_version(
    *,
    messages: list[LLMMessage] | None = None,
    invocation_params: InvocationParams | None = None,
    labels: list[str] | None = None,
    commit_message: str = "initial",
    provider: Any = LlmProvider.OPEN_AI,
    input_variable_format: Any = InputVariableFormat.F_STRING,
    model: str = "gpt-4o",
) -> PromptVersion:
    return PromptVersion(
        id="v1",
        prompt_id="p1",
        commit_hash="abc123",
        commit_message=commit_message,
        messages=messages if messages is not None else [],
        input_variable_format=input_variable_format,
        provider=provider,
        model=model,
        invocation_params=invocation_params,
        created_at=_NOW,
        created_by_user_id="user1",
        labels=labels,
    )


def _make_prompt_with_version(version: PromptVersion) -> PromptWithVersion:
    return PromptWithVersion(
        id="p1",
        name="my-prompt",
        description=None,
        space_id="s1",
        created_at=_NOW,
        updated_at=_NOW,
        created_by_user_id="user1",
        version=version,
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
            "ListPromptVersionsResponse",
            "PromptWithVersion",
            "ListPromptsResponse",
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
            ListPromptVersionsResponse,
            PromptWithVersion,
            ListPromptsResponse,
            ProviderParams,
        ],
    )
    def test_type_is_class(self, cls: type) -> None:
        assert isinstance(cls, type)


@pytest.mark.unit
class TestInvocationParamsStr:
    """Tests for InvocationParams.__str__."""

    def test_all_none_returns_empty_string(self) -> None:
        params = InvocationParams()
        assert str(params) == ""

    def test_single_field_set(self) -> None:
        params = InvocationParams(temperature=0.7)
        assert str(params) == "temperature=0.7"

    def test_multiple_fields_set(self) -> None:
        params = InvocationParams(temperature=0.5, max_tokens=100)
        result = str(params)
        assert "temperature=0.5" in result
        assert "max_tokens=100" in result
        assert result.count(",") == 1

    def test_none_fields_omitted(self) -> None:
        params = InvocationParams(temperature=0.9, max_tokens=None, top_p=0.95)
        result = str(params)
        assert "max_tokens" not in result
        assert "temperature=0.9" in result
        assert "top_p=0.95" in result

    def test_stop_sequences_shown(self) -> None:
        params = InvocationParams(stop=["END", "STOP"])
        assert "stop=" in str(params)


@pytest.mark.unit
class TestPromptVersionStr:
    """Tests for PromptVersion.__str__."""

    def test_basic_content_message(self) -> None:
        version = _make_version(
            messages=[
                LLMMessage(role=MessageRole.USER, content="Hello {name}")
            ],
            commit_message="add greeting",
        )
        result = str(version)
        assert "commit: add greeting" in result
        assert "USER: Hello {name}" in result
        assert "gpt-4o" in result
        assert "OPEN_AI" in result
        assert "F_STRING" in result

    def test_system_and_user_messages(self) -> None:
        version = _make_version(
            messages=[
                LLMMessage(role=MessageRole.SYSTEM, content="You are helpful."),
                LLMMessage(role=MessageRole.USER, content="Hi"),
            ]
        )
        result = str(version)
        assert "SYSTEM: You are helpful." in result
        assert "USER: Hi" in result

    def test_provider_as_plain_string(self) -> None:
        """Non-enum provider falls back to str() rather than .value."""
        # Use model_construct to bypass Pydantic enum validation so we can
        # exercise the isinstance-guard in __str__.
        version = _make_version()
        version = version.model_copy(update={"provider": "custom_provider"})
        assert "custom_provider" in str(version)

    def test_input_variable_format_as_plain_string(self) -> None:
        """Non-enum ivf falls back to str() rather than .value."""
        version = _make_version()
        version = version.model_copy(update={"input_variable_format": "jinja2"})
        assert "jinja2" in str(version)

    def test_labels_shown_when_present(self) -> None:
        version = _make_version(labels=["production", "staging"])
        result = str(version)
        assert "labels: production, staging" in result

    def test_labels_absent_when_none(self) -> None:
        version = _make_version(labels=None)
        assert "labels" not in str(version)

    def test_labels_absent_when_empty_list(self) -> None:
        version = _make_version(labels=[])
        assert "labels" not in str(version)

    def test_invocation_params_shown_when_set(self) -> None:
        params = InvocationParams(temperature=0.8)
        version = _make_version(invocation_params=params)
        assert "params: temperature=0.8" in str(version)

    def test_invocation_params_hidden_when_none(self) -> None:
        version = _make_version(invocation_params=None)
        assert "params" not in str(version)

    def test_invocation_params_hidden_when_all_none(self) -> None:
        """InvocationParams with all-None fields should not produce a params line."""
        params = InvocationParams()
        version = _make_version(invocation_params=params)
        assert "params" not in str(version)

    def test_empty_messages_shows_placeholder(self) -> None:
        version = _make_version(messages=[])
        assert "(no messages)" in str(version)

    def test_none_messages_shows_placeholder(self) -> None:
        version = _make_version(messages=None)
        result = str(version)
        assert "(no messages)" in result

    def test_message_with_tool_calls(self) -> None:
        fn = ToolCallFunction(name="search", arguments="{}")
        tc = ToolCall(type=ToolCallType.FUNCTION, function=fn)
        msg = LLMMessage(
            role=MessageRole.ASSISTANT, content=None, tool_calls=[tc]
        )
        version = _make_version(messages=[msg])
        result = str(version)
        assert "ASSISTANT" in result
        assert "search()" in result

    def test_message_with_tool_call_id(self) -> None:
        msg = LLMMessage(
            role=MessageRole.TOOL,
            content=None,
            tool_call_id="call_abc",
        )
        version = _make_version(messages=[msg])
        result = str(version)
        assert "tool response for call_abc" in result

    def test_message_with_no_content_no_tools(self) -> None:
        """Message with only role set should render as 'role:'."""
        msg = LLMMessage(role=MessageRole.ASSISTANT, content=None)
        version = _make_version(messages=[msg])
        result = str(version)
        assert "ASSISTANT:" in result

    def test_tool_call_with_no_function_attribute(self) -> None:
        """ToolCall whose function attribute is missing/None should show '?'."""
        fn = ToolCallFunction(name="do_thing", arguments="{}")
        tc = ToolCall(type=ToolCallType.FUNCTION, function=fn)
        msg = LLMMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[tc],
        )
        # Normal path: function.name = "do_thing"
        result = str(_make_version(messages=[msg]))
        assert "do_thing()" in result

    def test_commit_message_in_output(self) -> None:
        version = _make_version(commit_message="fix typo")
        assert "commit: fix typo" in str(version)

    def test_model_name_in_output(self) -> None:
        version = _make_version(model="claude-3-5-sonnet")
        assert "claude-3-5-sonnet" in str(version)

    def test_multiple_labels_joined_by_comma(self) -> None:
        version = _make_version(labels=["v1", "v2", "v3"])
        result = str(version)
        assert "v1, v2, v3" in result


@pytest.mark.unit
class TestPromptWithVersion:
    """Tests for PromptWithVersion."""

    def test_version_field_is_prompt_version_subclass(self) -> None:
        """The version field annotation should be our SDK PromptVersion."""
        annotation = PromptWithVersion.model_fields["version"].annotation
        assert annotation is PromptVersion

    def test_version_accessible_as_prompt_version(self) -> None:
        pv = _make_version(commit_message="test")
        pwv = _make_prompt_with_version(pv)
        assert isinstance(pwv.version, PromptVersion)

    def test_str_on_version_uses_sdk_repr(self) -> None:
        """PromptWithVersion.version should use the SDK __str__, not generated repr."""
        pv = _make_version(
            messages=[LLMMessage(role=MessageRole.USER, content="hi")],
            commit_message="my commit",
        )
        pwv = _make_prompt_with_version(pv)
        version_str = str(pwv.version)
        assert "commit: my commit" in version_str
        assert "USER: hi" in version_str


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


@pytest.mark.unit
class TestListPromptVersionsResponseToDF:
    """Contract tests: to_df() must return structured data, not display strings.

    Display formatting belongs in the CLI layer (ax/core/output.py).
    Callers rely on to_df() returning dicts/lists they can index into.
    """

    def _make_response(
        self, versions: list[PromptVersion]
    ) -> ListPromptVersionsResponse:
        from arize._generated.api_client.models.pagination_metadata import (
            PaginationMetadata,
        )

        return ListPromptVersionsResponse(
            prompt_versions=versions,
            pagination=PaginationMetadata(has_more=False, next_cursor=None),
        )

    def test_invocation_params_is_dict(self) -> None:
        """invocation_params must be a dict in the DataFrame, not a formatted string."""
        version = _make_version(
            invocation_params=InvocationParams(temperature=0.7, max_tokens=100)
        )
        df = self._make_response([version]).to_df()
        assert "invocation_params" in df.columns
        val = df["invocation_params"].iloc[0]
        assert isinstance(val, dict), f"expected dict, got {type(val)}: {val!r}"
        assert val["temperature"] == 0.7
        assert val["max_tokens"] == 100

    def test_invocation_params_none_column_dropped(self) -> None:
        """Column is absent when all versions have invocation_params=None."""
        df = self._make_response(
            [_make_version(invocation_params=None)]
        ).to_df()
        assert "invocation_params" not in df.columns

    def test_provider_params_is_dict(self) -> None:
        """provider_params must be a dict in the DataFrame, not a formatted string."""
        version = _make_version()
        version.provider_params = ProviderParams(anthropic_version="2023-06-01")
        df = self._make_response([version]).to_df()
        assert "provider_params" in df.columns
        val = df["provider_params"].iloc[0]
        assert isinstance(val, dict), f"expected dict, got {type(val)}: {val!r}"
        assert val["anthropic_version"] == "2023-06-01"

    def test_messages_is_list(self) -> None:
        """Messages must be a list of dicts in the DataFrame, not a formatted string."""
        version = _make_version(
            messages=[
                LLMMessage(role=MessageRole.SYSTEM, content="You are helpful"),
                LLMMessage(role=MessageRole.USER, content="Hello {{name}}"),
            ]
        )
        df = self._make_response([version]).to_df()
        assert "messages" in df.columns
        val = df["messages"].iloc[0]
        assert isinstance(val, list), f"expected list, got {type(val)}: {val!r}"
        assert len(val) == 2
