"""Unit tests for src/arize/integrations/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, create_autospec, patch

import pytest

from arize._generated import api_client as gen
from arize._generated.api_client import IntegrationsApi
from arize.integrations.client import IntegrationsClient
from arize.integrations.types import (
    IntegrationType,
    ListIntegrationsResponse,
)
from arize.utils.resolve import NotFoundError

# Base64 ID that decodes to "Integration:123" — passes is_resource_id()
_INTEGRATION_ID = "SW50ZWdyYXRpb246MTIz"


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock IntegrationsApi instance."""
    return create_autospec(IntegrationsApi, instance=True)


@pytest.fixture
def integrations_client(
    mock_sdk_config: Mock, mock_api: Mock
) -> IntegrationsClient:
    """Provide an IntegrationsClient with mocked internals."""
    with patch(
        "arize._generated.api_client.IntegrationsApi", return_value=mock_api
    ):
        return IntegrationsClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestIntegrationsClientInit:
    """Tests for IntegrationsClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.IntegrationsApi",
            return_value=mock_api,
        ):
            client = IntegrationsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to IntegrationsApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.IntegrationsApi"
        ) as mock_api_cls:
            IntegrationsClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestIntegrationsClientList:
    """Tests for IntegrationsClient.list()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            ListIntegrationsResponse,
            "model_validate",
            side_effect=lambda v, **kw: v,
        ):
            yield

    def test_list_with_space_id(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """list() should resolve a base64 resource ID space value to space_id."""
        integrations_client.list(
            integration_type=IntegrationType.LLM,
            name="my-integration",
            space="U3BhY2U6OTA1MDoxSmtS",
            limit=25,
            cursor="cursor-xyz",
        )

        mock_api.list_integrations.assert_called_once_with(
            type=IntegrationType.LLM,
            space_id="U3BhY2U6OTA1MDoxSmtS",
            space_name=None,
            name="my-integration",
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_with_space_name(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """list() should resolve a non-prefixed space value to space_name."""
        integrations_client.list(
            integration_type=IntegrationType.AGENT,
            space="my-space",
        )

        mock_api.list_integrations.assert_called_once_with(
            type=IntegrationType.AGENT,
            space_id=None,
            space_name="my-space",
            name=None,
            limit=50,
            cursor=None,
        )

    def test_list_defaults(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """list() with no arguments should request all types (type=None)."""
        integrations_client.list()

        mock_api.list_integrations.assert_called_once_with(
            type=None,
            space_id=None,
            space_name=None,
            name=None,
            limit=50,
            cursor=None,
        )

    def test_list_returns_wrapped_response(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """list() should propagate the (bypassed) validated response."""
        expected = Mock()
        mock_api.list_integrations.return_value = expected

        result = integrations_client.list(integration_type=IntegrationType.LLM)

        assert result is expected

    def test_list_emits_alpha_prerelease_warning(
        self,
        integrations_client: IntegrationsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        integrations_client.list(integration_type=IntegrationType.LLM)

        assert any(
            "ALPHA" in record.message and "integrations.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestIntegrationsClientGet:
    """Tests for IntegrationsClient.get()."""

    def test_get_calls_api_with_integration_id(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """get() by ID should not require a type and skip resolution."""
        integrations_client.get(integration=_INTEGRATION_ID)

        mock_api.get_integration.assert_called_once_with(
            integration_id=_INTEGRATION_ID
        )
        mock_api.list_integrations.assert_not_called()

    def test_get_by_name_without_type_raises(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """get() by name should require integration_type to resolve."""
        with pytest.raises(NotFoundError, match="integration_type"):
            integrations_client.get(integration="my-integration")

        mock_api.list_integrations.assert_not_called()
        mock_api.get_integration.assert_not_called()

    def test_get_unwraps_response(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """get() should unwrap the oneOf response to its actual_instance."""
        expected = Mock()
        mock_api.get_integration.return_value.actual_instance = expected

        result = integrations_client.get(
            integration=_INTEGRATION_ID, integration_type=IntegrationType.AGENT
        )

        assert result is expected

    def test_get_resolves_name_without_space(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """get() should resolve a name using type alone (space not required).

        Integration names are unique per ``(account, type)``, so ``type`` is
        sufficient to resolve a name to an ID; ``space`` is only a visibility
        filter.
        """
        match = Mock()
        match.actual_instance.name = "my-integration"
        match.actual_instance.id = _INTEGRATION_ID
        mock_api.list_integrations.return_value.integrations = [match]
        mock_api.list_integrations.return_value.pagination.next_cursor = None

        integrations_client.get(
            integration="my-integration",
            integration_type=IntegrationType.LLM,
        )

        # Resolution lists by type with no space filter, then fetches by ID.
        mock_api.list_integrations.assert_called_once_with(
            type=IntegrationType.LLM,
            space_id=None,
            space_name=None,
            name="my-integration",
            limit=100,
            cursor=None,
        )
        mock_api.get_integration.assert_called_once_with(
            integration_id=_INTEGRATION_ID
        )


def _created_llm_config(mock_api: Mock) -> object:
    """Extract the provider config passed through create_integration().

    Returns the ``actual_instance`` of the wrapped ``CreateLlmConfig`` — i.e.
    the concrete per-provider ``Create*Config`` object the client forwarded.
    """
    body = mock_api.create_integration.call_args.kwargs[
        "create_integration_request"
    ]
    inner = body.actual_instance
    assert isinstance(inner, gen.CreateLlmIntegrationRequest)
    assert inner.type == "LLM"
    assert isinstance(inner.config, gen.CreateLlmConfig)
    return inner.config.actual_instance


@pytest.mark.unit
class TestIntegrationsClientCreateLlm:
    """Tests for IntegrationsClient.create_llm() across all 7 providers."""

    def test_create_openai_builds_request(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_llm(OPEN_AI) should wrap and forward the OpenAI config."""
        config = gen.CreateOpenAiConfig(
            provider="OPEN_AI",
            api_key="sk-abc",
            is_function_calling_enabled=True,
        )
        integrations_client.create_llm(name="Prod OpenAI", config=config)

        mock_api.create_integration.assert_called_once()
        assert _created_llm_config(mock_api) is config
        body = mock_api.create_integration.call_args.kwargs[
            "create_integration_request"
        ]
        assert body.actual_instance.name == "Prod OpenAI"
        assert body.actual_instance.scopings is None

    def test_create_anthropic_builds_request(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_llm(ANTHROPIC) should forward the Anthropic config."""
        config = gen.CreateAnthropicConfig(
            provider="ANTHROPIC", api_key="sk-ant"
        )
        integrations_client.create_llm(name="Prod Anthropic", config=config)

        forwarded = _created_llm_config(mock_api)
        assert isinstance(forwarded, gen.CreateAnthropicConfig)
        assert forwarded.api_key == "sk-ant"

    def test_create_gemini_builds_request(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_llm(GEMINI) should forward the Gemini config."""
        config = gen.CreateGeminiConfig(provider="GEMINI", api_key="sk-gem")
        integrations_client.create_llm(name="Gemini", config=config)

        assert _created_llm_config(mock_api) is config

    def test_create_vertex_ai_builds_request(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_llm(VERTEX_AI) should forward the Vertex AI config."""
        config = gen.CreateVertexAiConfig(
            provider="VERTEX_AI",
            project_id="proj-1",
            location="us-central1",
            project_access_label="label",
        )
        integrations_client.create_llm(name="Vertex", config=config)

        forwarded = _created_llm_config(mock_api)
        assert isinstance(forwarded, gen.CreateVertexAiConfig)
        assert forwarded.location == "us-central1"

    def test_create_custom_builds_request(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_llm(CUSTOM) should forward the Custom config."""
        config = gen.CreateCustomConfig(
            provider="CUSTOM",
            base_url="https://custom.example.com",
            api_key="sk-custom",
            headers={"x-team": "ml"},
            model_names=["my-model"],
        )
        integrations_client.create_llm(name="Custom", config=config)

        forwarded = _created_llm_config(mock_api)
        assert isinstance(forwarded, gen.CreateCustomConfig)
        assert forwarded.base_url == "https://custom.example.com"

    def test_create_nvidia_nim_builds_request(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_llm(NVIDIA_NIM) should forward the NVIDIA NIM config."""
        config = gen.CreateNvidiaNimConfig(
            provider="NVIDIA_NIM",
            base_url="https://nim.example.com",
            is_default_models_enabled=True,
        )
        integrations_client.create_llm(name="NIM", config=config)

        forwarded = _created_llm_config(mock_api)
        assert isinstance(forwarded, gen.CreateNvidiaNimConfig)
        assert forwarded.is_default_models_enabled is True

    def test_create_bedrock_default_auth(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_llm(AWS_BEDROCK) with DEFAULT auth should forward correctly."""
        config = gen.CreateAwsBedrockConfig(
            provider="AWS_BEDROCK",
            auth=gen.CreateAwsBedrockAuth(
                actual_instance=gen.CreateAwsBedrockDefaultAuth(
                    auth_type="DEFAULT",
                    role_arn="arn:aws:iam::123:role/arize",
                    external_id="ext-1",
                )
            ),
            is_default_models_enabled=True,
        )
        integrations_client.create_llm(name="Bedrock Default", config=config)

        forwarded = _created_llm_config(mock_api)
        assert isinstance(forwarded, gen.CreateAwsBedrockConfig)
        auth = forwarded.auth.actual_instance
        assert isinstance(auth, gen.CreateAwsBedrockDefaultAuth)
        assert auth.role_arn == "arn:aws:iam::123:role/arize"

    def test_create_bedrock_bearer_token_auth(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_llm(AWS_BEDROCK) with BEARER_TOKEN auth should forward it."""
        config = gen.CreateAwsBedrockConfig(
            provider="AWS_BEDROCK",
            auth=gen.CreateAwsBedrockAuth(
                actual_instance=gen.CreateAwsBedrockBearerTokenAuth(
                    auth_type="BEARER_TOKEN",
                    api_key="bearer-xyz",
                )
            ),
            model_names=["anthropic.claude"],
        )
        integrations_client.create_llm(name="Bedrock Bearer", config=config)

        forwarded = _created_llm_config(mock_api)
        auth = forwarded.auth.actual_instance
        assert isinstance(auth, gen.CreateAwsBedrockBearerTokenAuth)
        assert auth.api_key == "bearer-xyz"

    def test_create_bedrock_proxy_with_headers_auth(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_llm(AWS_BEDROCK) with PROXY_WITH_HEADERS auth forwards it."""
        config = gen.CreateAwsBedrockConfig(
            provider="AWS_BEDROCK",
            auth=gen.CreateAwsBedrockAuth(
                actual_instance=gen.CreateAwsBedrockProxyWithHeadersAuth(
                    auth_type="PROXY_WITH_HEADERS",
                    base_url="https://proxy.example.com",
                    headers={"x-api": "v"},
                )
            ),
            model_names=["anthropic.claude"],
        )
        integrations_client.create_llm(name="Bedrock Proxy", config=config)

        forwarded = _created_llm_config(mock_api)
        auth = forwarded.auth.actual_instance
        assert isinstance(auth, gen.CreateAwsBedrockProxyWithHeadersAuth)
        assert auth.base_url == "https://proxy.example.com"

    def test_create_llm_accepts_prewrapped_config(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_llm() should accept a pre-wrapped CreateLlmConfig union."""
        inner = gen.CreateOpenAiConfig(provider="OPEN_AI", api_key="sk")
        wrapped = gen.CreateLlmConfig(actual_instance=inner)
        integrations_client.create_llm(name="n", config=wrapped)

        body = mock_api.create_integration.call_args.kwargs[
            "create_integration_request"
        ]
        assert body.actual_instance.config is wrapped

    def test_create_llm_forwards_scopings(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_llm() should forward provided scopings to the request."""
        scopings = [gen.IntegrationScopingRequest(space_id="sp-1")]
        config = gen.CreateOpenAiConfig(provider="OPEN_AI", api_key="sk")
        integrations_client.create_llm(
            name="n", config=config, scopings=scopings
        )

        body = mock_api.create_integration.call_args.kwargs[
            "create_integration_request"
        ]
        assert body.actual_instance.scopings == scopings

    def test_create_llm_unwraps_response(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_llm() should unwrap the oneOf response."""
        expected = Mock()
        mock_api.create_integration.return_value.actual_instance = expected

        config = gen.CreateOpenAiConfig(provider="OPEN_AI", api_key="sk")
        result = integrations_client.create_llm(name="n", config=config)

        assert result is expected


@pytest.mark.unit
class TestIntegrationsClientCreateAgent:
    """Tests for IntegrationsClient.create_agent()."""

    def test_create_agent_builds_request(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_agent() should build the agent config and request."""
        schema = {"type": "object"}
        with (
            patch(
                "arize._generated.api_client.CreateAgentConfig"
            ) as mock_cfg_cls,
            patch(
                "arize._generated.api_client.CreateAgentIntegrationRequest"
            ) as mock_req_cls,
            patch(
                "arize._generated.api_client.CreateIntegrationRequest"
            ) as mock_env_cls,
        ):
            integrations_client.create_agent(
                name="My Agent",
                endpoint="https://agent.example.com/run",
                input_schema=schema,
                description="desc",
                headers={"x-key": "v"},
            )

        mock_cfg_cls.assert_called_once_with(
            endpoint="https://agent.example.com/run",
            input_schema=schema,
            headers={"x-key": "v"},
            request_presets=None,
        )
        mock_req_cls.assert_called_once_with(
            type="AGENT",
            name="My Agent",
            description="desc",
            scopings=None,
            config=mock_cfg_cls.return_value,
        )
        mock_env_cls.assert_called_once_with(
            actual_instance=mock_req_cls.return_value
        )
        mock_api.create_integration.assert_called_once_with(
            create_integration_request=mock_env_cls.return_value
        )

    def test_create_agent_unwraps_response(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """create_agent() should unwrap the oneOf response."""
        expected = Mock()
        mock_api.create_integration.return_value.actual_instance = expected

        with (
            patch("arize._generated.api_client.CreateAgentConfig"),
            patch("arize._generated.api_client.CreateAgentIntegrationRequest"),
            patch("arize._generated.api_client.CreateIntegrationRequest"),
        ):
            result = integrations_client.create_agent(
                name="n",
                endpoint="https://x",
                input_schema={"type": "object"},
            )

        assert result is expected


@pytest.mark.unit
class TestIntegrationsClientUpdateLlm:
    """Tests for IntegrationsClient.update_llm()."""

    def test_update_only_sends_provided_fields(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_llm() should only send caller-provided envelope fields."""
        with (
            patch(
                "arize._generated.api_client.UpdateLlmIntegrationRequest"
            ) as mock_req_cls,
            patch("arize._generated.api_client.UpdateIntegrationRequest"),
        ):
            integrations_client.update_llm(
                integration=_INTEGRATION_ID,
                name="Updated",
            )

        mock_req_cls.assert_called_once_with(type="LLM", name="Updated")
        mock_api.update_integration.assert_called_once()

    def test_update_builds_config_for_api_key(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_llm() should build a config when api_key is provided."""
        with (
            patch(
                "arize._generated.api_client.UpdateLlmConfig"
            ) as mock_cfg_cls,
            patch(
                "arize._generated.api_client.UpdateLlmIntegrationRequest"
            ) as mock_req_cls,
            patch("arize._generated.api_client.UpdateIntegrationRequest"),
        ):
            integrations_client.update_llm(
                integration=_INTEGRATION_ID,
                api_key="sk-new",
                function_calling_enabled=False,
            )

        mock_cfg_cls.assert_called_once_with(
            api_key="sk-new",
            is_function_calling_enabled=False,
        )
        mock_req_cls.assert_called_once_with(
            type="LLM", config=mock_cfg_cls.return_value
        )

    def test_update_explicit_none_api_key_clears(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_llm() should forward explicit None api_key to clear it."""
        with (
            patch(
                "arize._generated.api_client.UpdateLlmConfig"
            ) as mock_cfg_cls,
            patch("arize._generated.api_client.UpdateLlmIntegrationRequest"),
            patch("arize._generated.api_client.UpdateIntegrationRequest"),
        ):
            integrations_client.update_llm(
                integration=_INTEGRATION_ID,
                api_key=None,
            )

        mock_cfg_cls.assert_called_once_with(api_key=None)

    def test_update_bedrock_auth(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_llm() should build a config from a replacement Bedrock auth."""
        auth = gen.CreateAwsBedrockAuth(
            actual_instance=gen.CreateAwsBedrockDefaultAuth(
                auth_type="DEFAULT", role_arn="arn:aws:iam::123:role/r"
            )
        )
        with (
            patch(
                "arize._generated.api_client.UpdateLlmConfig"
            ) as mock_cfg_cls,
            patch("arize._generated.api_client.UpdateLlmIntegrationRequest"),
            patch("arize._generated.api_client.UpdateIntegrationRequest"),
        ):
            integrations_client.update_llm(
                integration=_INTEGRATION_ID,
                auth=auth,
                model_names=["anthropic.claude"],
                is_default_models_enabled=True,
            )

        mock_cfg_cls.assert_called_once_with(
            auth=auth,
            is_default_models_enabled=True,
            model_names=["anthropic.claude"],
        )

    def test_update_custom_base_url_and_headers(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_llm() should forward base_url and headers for CUSTOM."""
        with (
            patch(
                "arize._generated.api_client.UpdateLlmConfig"
            ) as mock_cfg_cls,
            patch("arize._generated.api_client.UpdateLlmIntegrationRequest"),
            patch("arize._generated.api_client.UpdateIntegrationRequest"),
        ):
            integrations_client.update_llm(
                integration=_INTEGRATION_ID,
                base_url="https://new.example.com",
                headers={"x-team": "ml"},
            )

        mock_cfg_cls.assert_called_once_with(
            base_url="https://new.example.com",
            headers={"x-team": "ml"},
        )

    def test_update_nim_base_url_and_headers_clear(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_llm() should forward explicit None to clear base_url/headers."""
        with (
            patch(
                "arize._generated.api_client.UpdateLlmConfig"
            ) as mock_cfg_cls,
            patch("arize._generated.api_client.UpdateLlmIntegrationRequest"),
            patch("arize._generated.api_client.UpdateIntegrationRequest"),
        ):
            integrations_client.update_llm(
                integration=_INTEGRATION_ID,
                base_url=None,
                headers=None,
            )

        mock_cfg_cls.assert_called_once_with(base_url=None, headers=None)

    def test_update_vertex_ai_fields(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_llm() should forward Vertex AI project fields."""
        with (
            patch(
                "arize._generated.api_client.UpdateLlmConfig"
            ) as mock_cfg_cls,
            patch("arize._generated.api_client.UpdateLlmIntegrationRequest"),
            patch("arize._generated.api_client.UpdateIntegrationRequest"),
        ):
            integrations_client.update_llm(
                integration=_INTEGRATION_ID,
                project_id="proj-2",
                location="us-east1",
                project_access_label="label-2",
            )

        mock_cfg_cls.assert_called_once_with(
            project_id="proj-2",
            location="us-east1",
            project_access_label="label-2",
        )

    def test_update_no_fields_raises(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_llm() with no updatable fields should raise ValueError."""
        with pytest.raises(ValueError, match="At least one field"):
            integrations_client.update_llm(integration=_INTEGRATION_ID)

        mock_api.update_integration.assert_not_called()

    def test_update_replaces_scopings(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_llm() should include scopings when provided."""
        scopings = [Mock()]
        with (
            patch(
                "arize._generated.api_client.UpdateLlmIntegrationRequest"
            ) as mock_req_cls,
            patch("arize._generated.api_client.UpdateIntegrationRequest"),
        ):
            integrations_client.update_llm(
                integration=_INTEGRATION_ID,
                scopings=scopings,
            )

        mock_req_cls.assert_called_once_with(type="LLM", scopings=scopings)

    def test_update_llm_unwraps_response(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_llm() should unwrap the oneOf response."""
        expected = Mock()
        mock_api.update_integration.return_value.actual_instance = expected

        with (
            patch("arize._generated.api_client.UpdateLlmIntegrationRequest"),
            patch("arize._generated.api_client.UpdateIntegrationRequest"),
        ):
            result = integrations_client.update_llm(
                integration=_INTEGRATION_ID, name="x"
            )

        assert result is expected


@pytest.mark.unit
class TestIntegrationsClientUpdateAgent:
    """Tests for IntegrationsClient.update_agent()."""

    def test_update_builds_config_and_envelope(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_agent() should build config from provided config fields."""
        schema = {"type": "object"}
        with (
            patch(
                "arize._generated.api_client.UpdateAgentConfig"
            ) as mock_cfg_cls,
            patch(
                "arize._generated.api_client.UpdateAgentIntegrationRequest"
            ) as mock_req_cls,
            patch("arize._generated.api_client.UpdateIntegrationRequest"),
        ):
            integrations_client.update_agent(
                integration=_INTEGRATION_ID,
                name="Updated Agent",
                endpoint="https://new.example.com",
                input_schema=schema,
            )

        mock_cfg_cls.assert_called_once_with(
            endpoint="https://new.example.com",
            input_schema=schema,
        )
        mock_req_cls.assert_called_once_with(
            type="AGENT",
            name="Updated Agent",
            config=mock_cfg_cls.return_value,
        )

    def test_update_explicit_none_clears_nullable_fields(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_agent() should forward explicit None for description/headers."""
        with (
            patch(
                "arize._generated.api_client.UpdateAgentConfig"
            ) as mock_cfg_cls,
            patch(
                "arize._generated.api_client.UpdateAgentIntegrationRequest"
            ) as mock_req_cls,
            patch("arize._generated.api_client.UpdateIntegrationRequest"),
        ):
            integrations_client.update_agent(
                integration=_INTEGRATION_ID,
                description=None,
                headers=None,
            )

        mock_cfg_cls.assert_called_once_with(headers=None)
        mock_req_cls.assert_called_once_with(
            type="AGENT",
            description=None,
            config=mock_cfg_cls.return_value,
        )

    def test_update_no_fields_raises(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_agent() with no updatable fields should raise ValueError."""
        with pytest.raises(ValueError, match="At least one field"):
            integrations_client.update_agent(integration=_INTEGRATION_ID)

        mock_api.update_integration.assert_not_called()

    def test_update_agent_replaces_scopings_and_presets(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """update_agent() should include scopings and request_presets when given."""
        presets = [Mock()]
        scopings = [Mock()]
        with (
            patch(
                "arize._generated.api_client.UpdateAgentConfig"
            ) as mock_cfg_cls,
            patch(
                "arize._generated.api_client.UpdateAgentIntegrationRequest"
            ) as mock_req_cls,
            patch("arize._generated.api_client.UpdateIntegrationRequest"),
        ):
            integrations_client.update_agent(
                integration=_INTEGRATION_ID,
                request_presets=presets,
                scopings=scopings,
            )

        mock_cfg_cls.assert_called_once_with(request_presets=presets)
        mock_req_cls.assert_called_once_with(
            type="AGENT",
            scopings=scopings,
            config=mock_cfg_cls.return_value,
        )


@pytest.mark.unit
class TestIntegrationsClientDelete:
    """Tests for IntegrationsClient.delete()."""

    def test_delete_calls_api_with_integration_id(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """delete() by ID should not require a type and skip resolution."""
        integrations_client.delete(integration=_INTEGRATION_ID)

        mock_api.delete_integration.assert_called_once_with(
            integration_id=_INTEGRATION_ID
        )
        mock_api.list_integrations.assert_not_called()

    def test_delete_by_name_without_type_raises(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """delete() by name should require integration_type to resolve."""
        with pytest.raises(NotFoundError, match="integration_type"):
            integrations_client.delete(integration="my-integration")

        mock_api.list_integrations.assert_not_called()
        mock_api.delete_integration.assert_not_called()

    def test_delete_returns_none(
        self, integrations_client: IntegrationsClient, mock_api: Mock
    ) -> None:
        """delete() should return None on success."""
        result = integrations_client.delete(
            integration=_INTEGRATION_ID, integration_type=IntegrationType.LLM
        )

        assert result is None
