"""Public types for the integrations subdomain."""

from pydantic import BaseModel, ConfigDict, field_validator

from arize._generated.api_client.models.agent_config import AgentConfig
from arize._generated.api_client.models.agent_integration import (
    AgentIntegration,
)
from arize._generated.api_client.models.agent_request_preset import (
    AgentRequestPreset,
)
from arize._generated.api_client.models.anthropic_config import AnthropicConfig
from arize._generated.api_client.models.aws_bedrock_auth import AwsBedrockAuth
from arize._generated.api_client.models.aws_bedrock_bearer_token_auth import (
    AwsBedrockBearerTokenAuth,
)
from arize._generated.api_client.models.aws_bedrock_config import (
    AwsBedrockConfig,
)
from arize._generated.api_client.models.aws_bedrock_default_auth import (
    AwsBedrockDefaultAuth,
)
from arize._generated.api_client.models.aws_bedrock_proxy_with_headers_auth import (
    AwsBedrockProxyWithHeadersAuth,
)
from arize._generated.api_client.models.create_agent_request_preset_input import (
    CreateAgentRequestPresetInput,
)
from arize._generated.api_client.models.create_anthropic_config import (
    CreateAnthropicConfig,
)
from arize._generated.api_client.models.create_aws_bedrock_auth import (
    CreateAwsBedrockAuth,
)
from arize._generated.api_client.models.create_aws_bedrock_bearer_token_auth import (
    CreateAwsBedrockBearerTokenAuth,
)
from arize._generated.api_client.models.create_aws_bedrock_config import (
    CreateAwsBedrockConfig,
)
from arize._generated.api_client.models.create_aws_bedrock_default_auth import (
    CreateAwsBedrockDefaultAuth,
)
from arize._generated.api_client.models.create_aws_bedrock_proxy_with_headers_auth import (
    CreateAwsBedrockProxyWithHeadersAuth,
)
from arize._generated.api_client.models.create_custom_config import (
    CreateCustomConfig,
)
from arize._generated.api_client.models.create_gemini_config import (
    CreateGeminiConfig,
)
from arize._generated.api_client.models.create_lite_llm_config import (
    CreateLiteLlmConfig,
)
from arize._generated.api_client.models.create_llm_config import CreateLlmConfig
from arize._generated.api_client.models.create_nvidia_nim_config import (
    CreateNvidiaNimConfig,
)
from arize._generated.api_client.models.create_open_ai_config import (
    CreateOpenAiConfig,
)
from arize._generated.api_client.models.create_vertex_ai_config import (
    CreateVertexAiConfig,
)
from arize._generated.api_client.models.custom_config import CustomConfig
from arize._generated.api_client.models.gemini_config import GeminiConfig
from arize._generated.api_client.models.integration import Integration
from arize._generated.api_client.models.integration_scoping import (
    IntegrationScoping,
)
from arize._generated.api_client.models.integration_type import IntegrationType
from arize._generated.api_client.models.lite_llm_config import LiteLlmConfig
from arize._generated.api_client.models.llm_config import LlmConfig
from arize._generated.api_client.models.llm_integration import LlmIntegration
from arize._generated.api_client.models.llm_integration_provider import (
    LlmIntegrationProvider,
)
from arize._generated.api_client.models.nvidia_nim_config import NvidiaNimConfig
from arize._generated.api_client.models.open_ai_config import OpenAiConfig
from arize._generated.api_client.models.pagination_metadata import (
    PaginationMetadata,
)
from arize._generated.api_client.models.update_agent_request_preset_input import (
    UpdateAgentRequestPresetInput,
)
from arize._generated.api_client.models.update_llm_config import UpdateLlmConfig
from arize._generated.api_client.models.vertex_ai_config import VertexAiConfig


class ListIntegrationsResponse(BaseModel):
    """SDK view of the generated list response with each ``Integration`` unwrapped.

    The ``integrations`` field contains the concrete inner types
    (:class:`AgentIntegration` or :class:`LlmIntegration`) instead of the
    oneOf wrapper :class:`Integration`.
    """

    integrations: list[AgentIntegration | LlmIntegration]
    pagination: PaginationMetadata

    model_config = ConfigDict(from_attributes=True)

    @field_validator("integrations", mode="before")
    @classmethod
    def _coerce_integrations(
        cls, v: object
    ) -> list[AgentIntegration | LlmIntegration]:
        result = []
        for item in v:  # type: ignore[attr-defined]
            if isinstance(item, Integration):
                if item.actual_instance is None:
                    raise ValueError(
                        "Integration wrapper has actual_instance=None"
                    )
                item = item.actual_instance
            result.append(item)
        return result


__all__ = [
    "AgentConfig",
    "AgentIntegration",
    "AgentRequestPreset",
    "AnthropicConfig",
    "AwsBedrockAuth",
    "AwsBedrockBearerTokenAuth",
    "AwsBedrockConfig",
    "AwsBedrockDefaultAuth",
    "AwsBedrockProxyWithHeadersAuth",
    "CreateAgentRequestPresetInput",
    "CreateAnthropicConfig",
    "CreateAwsBedrockAuth",
    "CreateAwsBedrockBearerTokenAuth",
    "CreateAwsBedrockConfig",
    "CreateAwsBedrockDefaultAuth",
    "CreateAwsBedrockProxyWithHeadersAuth",
    "CreateCustomConfig",
    "CreateGeminiConfig",
    "CreateLiteLlmConfig",
    "CreateLlmConfig",
    "CreateNvidiaNimConfig",
    "CreateOpenAiConfig",
    "CreateVertexAiConfig",
    "CustomConfig",
    "GeminiConfig",
    "IntegrationScoping",
    "IntegrationType",
    "ListIntegrationsResponse",
    "LiteLlmConfig",
    "LlmConfig",
    "LlmIntegration",
    "LlmIntegrationProvider",
    "NvidiaNimConfig",
    "OpenAiConfig",
    "PaginationMetadata",
    "UpdateAgentRequestPresetInput",
    "UpdateLlmConfig",
    "VertexAiConfig",
]
