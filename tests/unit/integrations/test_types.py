"""Tests for arize.integrations.types public re-exports and unwrapping."""

from __future__ import annotations

from enum import Enum
from unittest.mock import Mock

import pytest

import arize.integrations.types as types_module
from arize._generated.api_client.models.agent_integration import (
    AgentIntegration,
)
from arize._generated.api_client.models.integration import Integration
from arize._generated.api_client.models.llm_integration import LlmIntegration
from arize._generated.api_client.models.pagination_metadata import (
    PaginationMetadata,
)
from arize.integrations.types import (
    IntegrationType,
    ListIntegrationsResponse,
    LlmIntegrationProvider,
)


@pytest.mark.unit
class TestIntegrationsTypes:
    """Tests for the integrations types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "IntegrationType" in types_module.__all__
        assert "AgentIntegration" in types_module.__all__
        assert "LlmIntegration" in types_module.__all__
        assert "ListIntegrationsResponse" in types_module.__all__
        assert "AgentRequestPreset" in types_module.__all__
        assert "CreateAgentRequestPresetInput" in types_module.__all__
        assert "UpdateAgentRequestPresetInput" in types_module.__all__

    def test_all_read_configs_exported(self) -> None:
        """Every provider read config type should be re-exported."""
        for name in (
            "OpenAiConfig",
            "AnthropicConfig",
            "GeminiConfig",
            "AwsBedrockConfig",
            "CustomConfig",
            "VertexAiConfig",
            "NvidiaNimConfig",
            "LiteLlmConfig",
        ):
            assert name in types_module.__all__

    def test_all_create_configs_and_auth_exported(self) -> None:
        """Every create config type plus Bedrock auth variants are exported."""
        for name in (
            "CreateOpenAiConfig",
            "CreateAnthropicConfig",
            "CreateGeminiConfig",
            "CreateAwsBedrockConfig",
            "CreateCustomConfig",
            "CreateVertexAiConfig",
            "CreateNvidiaNimConfig",
            "CreateLiteLlmConfig",
            "CreateLlmConfig",
            "UpdateLlmConfig",
            "AwsBedrockAuth",
            "AwsBedrockDefaultAuth",
            "AwsBedrockBearerTokenAuth",
            "AwsBedrockProxyWithHeadersAuth",
            "CreateAwsBedrockAuth",
            "CreateAwsBedrockDefaultAuth",
            "CreateAwsBedrockBearerTokenAuth",
            "CreateAwsBedrockProxyWithHeadersAuth",
        ):
            assert name in types_module.__all__

    def test_integration_type_is_enum(self) -> None:
        assert issubclass(IntegrationType, Enum)

    def test_llm_integration_provider_is_enum(self) -> None:
        assert issubclass(LlmIntegrationProvider, Enum)

    def test_list_integrations_response_is_class(self) -> None:
        assert isinstance(ListIntegrationsResponse, type)


def _agent_integration(name: str = "agent-1") -> AgentIntegration:
    return AgentIntegration.from_dict(
        {
            "id": "id-agent",
            "type": "AGENT",
            "name": name,
            "description": None,
            "scopings": [],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "created_by_user_id": "u1",
            "config": {
                "endpoint": "https://a.example.com",
                "has_headers": False,
                "input_schema": {"type": "object"},
                "request_presets": [],
            },
        }
    )


def _llm_integration(name: str = "llm-1") -> LlmIntegration:
    return LlmIntegration.from_dict(
        {
            "id": "id-llm",
            "type": "LLM",
            "name": name,
            "scopings": [],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "created_by_user_id": "u1",
            "config": {
                "provider": "OPEN_AI",
                "has_api_key": True,
                "is_function_calling_enabled": True,
            },
        }
    )


@pytest.mark.unit
class TestListIntegrationsResponseUnwrap:
    """Tests for ListIntegrationsResponse oneOf unwrapping."""

    def test_unwraps_wrapped_integrations(self) -> None:
        """Each wrapped Integration should be replaced by its actual_instance."""
        agent = _agent_integration()
        llm = _llm_integration()
        source = Mock()
        source.integrations = [
            Integration(actual_instance=agent),
            Integration(actual_instance=llm),
        ]
        source.pagination = PaginationMetadata(has_more=False, next_cursor=None)

        result = ListIntegrationsResponse.model_validate(
            source, from_attributes=True
        )

        assert result.integrations == [agent, llm]
        assert isinstance(result.integrations[0], AgentIntegration)
        assert isinstance(result.integrations[1], LlmIntegration)

    def test_accepts_already_unwrapped_items(self) -> None:
        """Concrete (non-wrapped) items should pass through unchanged."""
        llm = _llm_integration()
        source = Mock()
        source.integrations = [llm]
        source.pagination = PaginationMetadata(has_more=True, next_cursor="abc")

        result = ListIntegrationsResponse.model_validate(
            source, from_attributes=True
        )

        assert result.integrations == [llm]

    def test_raises_when_actual_instance_is_none(self) -> None:
        """A wrapper with actual_instance=None should raise a ValueError."""
        empty = Integration.model_construct(actual_instance=None)
        source = Mock()
        source.integrations = [empty]
        source.pagination = PaginationMetadata(has_more=False, next_cursor=None)

        with pytest.raises(ValueError, match="actual_instance=None"):
            ListIntegrationsResponse.model_validate(
                source, from_attributes=True
            )
