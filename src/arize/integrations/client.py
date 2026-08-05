"""Client implementation for managing integrations in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from arize._utils import unwrap_oneof
from arize.constants.config import DEFAULT_LIST_LIMIT
from arize.integrations.types import (
    CreateAnthropicConfig,
    CreateAwsBedrockConfig,
    CreateCustomConfig,
    CreateGeminiConfig,
    CreateLlmConfig,
    CreateNvidiaNimConfig,
    CreateOpenAiConfig,
    CreateVertexAiConfig,
    IntegrationType,
    ListIntegrationsResponse,
)
from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import (
    _find_integration_id,
    _resolve_resource,
)
from arize.utils.unset import _UNSET, UNSET, is_provided

if TYPE_CHECKING:
    # builtins is needed to use builtins.list in type annotations because
    # the class has a list() method that shadows the built-in list type
    import builtins

    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration
    from arize.integrations.types import (
        AgentIntegration,
        CreateAgentRequestPresetInput,
        CreateAwsBedrockAuth,
        IntegrationScoping,
        LlmIntegration,
        UpdateAgentRequestPresetInput,
    )

# The provider-discriminated config accepted by ``create_llm``. Callers
# construct the generated per-provider config for the provider they want
# (all 7 are supported); a pre-wrapped ``CreateLlmConfig`` is also accepted.
# Declared at runtime (not under TYPE_CHECKING) so downstream consumers such as
# the ax CLI can import and reference it.
CreateLlmConfigInput = (
    CreateOpenAiConfig
    | CreateAnthropicConfig
    | CreateGeminiConfig
    | CreateAwsBedrockConfig
    | CreateCustomConfig
    | CreateVertexAiConfig
    | CreateNvidiaNimConfig
    | CreateLlmConfig
)

logger = logging.getLogger(__name__)


class IntegrationsClient:
    """Client for managing Arize integrations (LLM and agent).

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The integrations client is a thin wrapper around the generated REST API
    client, using the shared generated API client owned by
    :class:`arize.config.SDKConfiguration`.

    Integrations are polymorphic: ``LLM`` integrations configure a model
    provider (``OPEN_AI``, ``ANTHROPIC``, ``GEMINI``, ``AWS_BEDROCK``,
    ``CUSTOM``, ``VERTEX_AI``, or ``NVIDIA_NIM``), while ``AGENT`` integrations
    connect a customer-hosted agent exposed at an HTTP endpoint. The integration
    :class:`~arize.integrations.types.IntegrationType` selects the config shape.
    """

    def __init__(
        self, *, sdk_config: SDKConfiguration, generated_client: ApiClient
    ) -> None:
        """
        Args:
            sdk_config: Resolved SDK configuration.
            generated_client: Shared generated API client instance.
        """  # noqa: D205, D212
        self._sdk_config = sdk_config

        # Import at runtime to keep the module lazy-loaded
        from arize._generated import api_client as gen

        self._api = gen.IntegrationsApi(generated_client)

    @prerelease_endpoint(key="integrations.list", stage=ReleaseStage.ALPHA)
    def list(
        self,
        *,
        integration_type: IntegrationType | None = None,
        name: str | None = None,
        space: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> ListIntegrationsResponse:
        """List integrations the user has access to.

        When *integration_type* is omitted, integrations of every type are
        returned in one merged list; each item carries its type. Integrations
        are returned in descending creation order (most recently created
        first).

        Args:
            integration_type: Optional filter to a single integration type
                (:attr:`~arize.integrations.types.IntegrationType.LLM` or
                :attr:`~arize.integrations.types.IntegrationType.AGENT`).
            name: Optional case-insensitive substring filter on the integration name.
            space: Optional space filter. If the value is a base64-encoded resource ID it is
                treated as a space ID; otherwise it is used as a case-insensitive
                substring filter on the space name.
            limit: Maximum number of integrations to return. The server may enforce
                an upper bound (max 100).
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A response object with the integrations and pagination information.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/429).
        """
        resolved_space = _resolve_resource(space)
        result = self._api.list_integrations(
            type=integration_type,
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=name,
            limit=limit,
            cursor=cursor,
        )
        return ListIntegrationsResponse.model_validate(
            result, from_attributes=True
        )

    @prerelease_endpoint(key="integrations.get", stage=ReleaseStage.ALPHA)
    def get(
        self,
        *,
        integration: str,
        integration_type: IntegrationType | None = None,
        space: str | None = None,
    ) -> AgentIntegration | LlmIntegration:
        """Get an integration by ID or name.

        Args:
            integration: Integration ID or name. If a name is provided,
                *integration_type* is used to resolve it (and *space* if given).
            integration_type: The integration type used to resolve
                *integration* by name. Names are only unique per
                ``(account, type)``, so this is required when *integration* is
                a name; it is ignored when *integration* is an ID.
            space: Optional space ID or name. This is only a visibility
                filter, not required to resolve a name.

        Returns:
            The concrete integration object (:class:`AgentIntegration` or
            :class:`LlmIntegration`).

        Raises:
            NotFoundError: If *integration* is a name and *integration_type*
                is not provided, or the name cannot be resolved.
            ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        integration_id = _find_integration_id(
            api=self._api,
            integration=integration,
            integration_type=integration_type,
            space=space,
        )
        result = self._api.get_integration(integration_id=integration_id)
        return unwrap_oneof(result)  # type: ignore[return-value]

    @prerelease_endpoint(key="integrations.create", stage=ReleaseStage.ALPHA)
    def create_llm(
        self,
        *,
        name: str,
        config: CreateLlmConfigInput,
        scopings: builtins.list[IntegrationScoping] | None = None,
    ) -> LlmIntegration:
        """Create an LLM integration.

        LLM integrations configure access to a model provider for use within
        the Arize platform. All 7 providers are supported; construct the
        matching generated config for the ``config`` argument:

        - ``OPEN_AI`` — :class:`~arize.integrations.types.CreateOpenAiConfig`
        - ``ANTHROPIC`` — :class:`~arize.integrations.types.CreateAnthropicConfig`
        - ``GEMINI`` — :class:`~arize.integrations.types.CreateGeminiConfig`
        - ``AWS_BEDROCK`` — :class:`~arize.integrations.types.CreateAwsBedrockConfig`
          (nests a :class:`~arize.integrations.types.CreateAwsBedrockAuth`:
          DEFAULT, BEARER_TOKEN, or PROXY_WITH_HEADERS)
        - ``CUSTOM`` — :class:`~arize.integrations.types.CreateCustomConfig`
        - ``VERTEX_AI`` — :class:`~arize.integrations.types.CreateVertexAiConfig`
        - ``NVIDIA_NIM`` — :class:`~arize.integrations.types.CreateNvidiaNimConfig`

        Integration names must be unique within the account for the ``LLM`` type.

        Args:
            name: Integration name (must be unique within the account per type).
            config: The provider-specific config. Accepts any of the 7 generated
                per-provider ``Create*Config`` objects, or a pre-wrapped
                :class:`~arize.integrations.types.CreateLlmConfig` union.
            scopings: Visibility scoping rules. Defaults to account-wide if omitted.

        Returns:
            The created LLM integration.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/409/422/429).
        """
        from arize._generated import api_client as gen

        llm_config = (
            config
            if isinstance(config, gen.CreateLlmConfig)
            else gen.CreateLlmConfig(actual_instance=config)
        )
        body = gen.CreateIntegrationRequest(
            actual_instance=gen.CreateLlmIntegrationRequest(
                type=IntegrationType.LLM.value,
                name=name,
                scopings=scopings,
                config=llm_config,
            )
        )
        result = self._api.create_integration(create_integration_request=body)
        return unwrap_oneof(result)  # type: ignore[return-value]

    @prerelease_endpoint(key="integrations.create", stage=ReleaseStage.ALPHA)
    def create_agent(
        self,
        *,
        name: str,
        endpoint: str,
        input_schema: dict[str, Any],
        description: str | None = None,
        headers: dict[str, str] | None = None,
        request_presets: builtins.list[CreateAgentRequestPresetInput]
        | None = None,
        scopings: builtins.list[IntegrationScoping] | None = None,
    ) -> AgentIntegration:
        """Create an agent integration.

        Agent integrations connect a customer-hosted agent exposed at an HTTPS
        endpoint. Integration names must be unique within the account for the
        ``AGENT`` type.

        Args:
            name: Integration name (must be unique within the account per type).
            endpoint: HTTPS endpoint URL Arize calls for replay. Validated
                server-side for SSRF (must resolve to a public address).
            input_schema: JSON Schema (Draft-07) the endpoint's request body
                conforms to.
            description: Optional human-readable description of the integration.
            headers: Optional custom headers to include in requests. Encrypted
                at rest and never returned in responses.
            request_presets: Optional initial named request presets.
            scopings: Visibility scoping rules. Defaults to account-wide if omitted.

        Returns:
            The created agent integration.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/409/422/429).
        """
        from arize._generated import api_client as gen

        body = gen.CreateIntegrationRequest(
            actual_instance=gen.CreateAgentIntegrationRequest(
                type=IntegrationType.AGENT.value,
                name=name,
                description=description,
                scopings=scopings,
                config=gen.CreateAgentConfig(
                    endpoint=endpoint,
                    input_schema=input_schema,
                    headers=headers,
                    request_presets=request_presets,
                ),
            )
        )
        result = self._api.create_integration(create_integration_request=body)
        return unwrap_oneof(result)  # type: ignore[return-value]

    @prerelease_endpoint(key="integrations.update", stage=ReleaseStage.ALPHA)
    def update_llm(
        self,
        *,
        integration: str,
        space: str | None = None,
        name: str | UNSET = _UNSET,
        api_key: str | None | UNSET = _UNSET,
        function_calling_enabled: bool | UNSET = _UNSET,
        auth: CreateAwsBedrockAuth | UNSET = _UNSET,
        base_url: str | None | UNSET = _UNSET,
        headers: dict[str, str] | None | UNSET = _UNSET,
        is_default_models_enabled: bool | UNSET = _UNSET,
        model_names: builtins.list[str] | UNSET = _UNSET,
        project_id: str | UNSET = _UNSET,
        location: str | UNSET = _UNSET,
        project_access_label: str | UNSET = _UNSET,
        scopings: builtins.list[IntegrationScoping] | UNSET = _UNSET,
    ) -> LlmIntegration:
        """Update an LLM integration by ID or name.

        At least one updatable field must be provided; otherwise a
        ``ValueError`` is raised (the server rejects type-only PATCHes).
        Only the fields you pass are sent to the server; omitted fields are
        left unchanged. The provider is immutable. The config fields map to the
        flat ``UpdateLlmConfig`` and are provider-conditional; the server
        rejects fields that do not apply to the stored provider with a 422, so
        pass only the fields valid for that provider:

        - ``api_key``, ``function_calling_enabled`` — all providers except
          ``AWS_BEDROCK`` and ``VERTEX_AI``.
        - ``auth`` — ``AWS_BEDROCK`` only; replaces the stored auth wholesale.
        - ``base_url``, ``headers`` — ``CUSTOM`` and ``NVIDIA_NIM`` only.
        - ``is_default_models_enabled``, ``model_names`` — ``AWS_BEDROCK``,
          ``CUSTOM``, and ``NVIDIA_NIM`` only.
        - ``project_id``, ``location``, ``project_access_label`` —
          ``VERTEX_AI`` only.

        Nullable fields (``api_key``, ``base_url``, ``headers``) accept an
        explicit ``None`` to clear them (omit to keep unchanged).

        Args:
            integration: Integration ID or name. If a name is provided, it is
                resolved using the ``LLM`` type (and *space* if given).
            space: Optional space ID or name. Integration names are unique per
                ``(account, type)``, so this is only a visibility filter, not
                required to resolve a name.
            name: New integration name. Must be unique within the account per type.
            api_key: New API key. Pass ``None`` to clear the existing key.
            function_calling_enabled: Updated function calling flag.
            auth: Replacement AWS Bedrock auth
                (:class:`~arize.integrations.types.CreateAwsBedrockAuth`).
            base_url: New endpoint URL. Pass ``None`` to clear (``NVIDIA_NIM``;
                ``CUSTOM`` rejects ``None`` with a 422).
            headers: Replacement custom request headers as a name-to-value map.
                Pass ``None`` to clear all headers.
            is_default_models_enabled: Toggle Arize's default model catalog.
            model_names: Replacement custom model list.
            project_id: New Vertex AI GCP project ID.
            location: New Vertex AI GCP region.
            project_access_label: New Vertex AI project-access label.
            scopings: Replacement visibility scoping rules (replaces all existing).

        Returns:
            The updated LLM integration.

        Raises:
            ValueError: If no updatable field is provided.
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/404/409/422/429).
        """
        from arize._generated import api_client as gen

        config_kwargs: dict[str, Any] = {
            k: v
            for k, v in (
                ("api_key", api_key),
                ("is_function_calling_enabled", function_calling_enabled),
                ("auth", auth),
                ("base_url", base_url),
                ("headers", headers),
                ("is_default_models_enabled", is_default_models_enabled),
                ("model_names", model_names),
                ("project_id", project_id),
                ("location", location),
                ("project_access_label", project_access_label),
            )
            if is_provided(v)
        }
        envelope_kwargs: dict[str, Any] = {"type": IntegrationType.LLM.value}
        if is_provided(name):
            envelope_kwargs["name"] = name
        if is_provided(scopings):
            envelope_kwargs["scopings"] = scopings
        if config_kwargs:
            envelope_kwargs["config"] = gen.UpdateLlmConfig(**config_kwargs)

        # The API rejects type-only PATCHes; reject empty updates locally so
        # callers get a clear error instead of an opaque 422.
        if (
            not is_provided(name)
            and not is_provided(scopings)
            and not config_kwargs
        ):
            raise ValueError(
                "At least one field must be provided to update the "
                "integration (name, scopings, or a config field)."
            )

        integration_id = _find_integration_id(
            api=self._api,
            integration=integration,
            integration_type=IntegrationType.LLM,
            space=space,
        )
        body = gen.UpdateIntegrationRequest(
            actual_instance=gen.UpdateLlmIntegrationRequest(**envelope_kwargs)
        )
        result = self._api.update_integration(
            integration_id=integration_id,
            update_integration_request=body,
        )
        return unwrap_oneof(result)  # type: ignore[return-value]

    @prerelease_endpoint(key="integrations.update", stage=ReleaseStage.ALPHA)
    def update_agent(
        self,
        *,
        integration: str,
        space: str | None = None,
        name: str | UNSET = _UNSET,
        description: str | None | UNSET = _UNSET,
        endpoint: str | UNSET = _UNSET,
        input_schema: dict[str, Any] | UNSET = _UNSET,
        headers: dict[str, str] | None | UNSET = _UNSET,
        request_presets: builtins.list[UpdateAgentRequestPresetInput]
        | UNSET = _UNSET,
        scopings: builtins.list[IntegrationScoping] | UNSET = _UNSET,
    ) -> AgentIntegration:
        """Update an agent integration by ID or name.

        At least one updatable field must be provided; otherwise a
        ``ValueError`` is raised (the server rejects type-only PATCHes).
        Only the fields you pass are sent to the server; omitted fields are
        left unchanged. Collection fields (``headers``, ``request_presets``,
        ``scopings``) replace the existing values when provided. To clear
        nullable fields (``description``, ``headers``), pass ``None``.

        Args:
            integration: Integration ID or name. If a name is provided, it is
                resolved using the ``AGENT`` type (and *space* if given).
            space: Optional space ID or name. Integration names are unique per
                ``(account, type)``, so this is only a visibility filter, not
                required to resolve a name.
            name: New integration name. Must be unique within the account per type.
            description: New description. Pass ``None`` to clear.
            endpoint: New HTTPS endpoint URL.
            input_schema: New JSON Schema for the request payload shape.
            headers: Replacement custom headers. Pass ``None`` (or ``{}``) to
                clear all headers.
            request_presets: Replacement request presets, matched by ``name``.
            scopings: Replacement visibility scoping rules (replaces all existing).

        Returns:
            The updated agent integration.

        Raises:
            ValueError: If no updatable field is provided.
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/404/409/422/429).
        """
        from arize._generated import api_client as gen

        config_kwargs: dict[str, Any] = {
            k: v
            for k, v in (
                ("endpoint", endpoint),
                ("input_schema", input_schema),
                ("headers", headers),
                ("request_presets", request_presets),
            )
            if is_provided(v)
        }
        envelope_kwargs: dict[str, Any] = {"type": IntegrationType.AGENT.value}
        if is_provided(name):
            envelope_kwargs["name"] = name
        if is_provided(description):
            envelope_kwargs["description"] = description
        if is_provided(scopings):
            envelope_kwargs["scopings"] = scopings
        if config_kwargs:
            envelope_kwargs["config"] = gen.UpdateAgentConfig(**config_kwargs)

        # The API rejects type-only PATCHes; reject empty updates locally so
        # callers get a clear error instead of an opaque 422.
        if (
            not is_provided(name)
            and not is_provided(description)
            and not is_provided(scopings)
            and not config_kwargs
        ):
            raise ValueError(
                "At least one field must be provided to update the "
                "integration (name, description, scopings, or a config field)."
            )

        integration_id = _find_integration_id(
            api=self._api,
            integration=integration,
            integration_type=IntegrationType.AGENT,
            space=space,
        )
        body = gen.UpdateIntegrationRequest(
            actual_instance=gen.UpdateAgentIntegrationRequest(**envelope_kwargs)
        )
        result = self._api.update_integration(
            integration_id=integration_id,
            update_integration_request=body,
        )
        return unwrap_oneof(result)  # type: ignore[return-value]

    @prerelease_endpoint(key="integrations.delete", stage=ReleaseStage.ALPHA)
    def delete(
        self,
        *,
        integration: str,
        integration_type: IntegrationType | None = None,
        space: str | None = None,
    ) -> None:
        """Delete an integration by ID or name.

        This operation is irreversible.

        Args:
            integration: Integration ID or name. If a name is provided,
                *integration_type* is used to resolve it (and *space* if given).
            integration_type: The integration type used to resolve
                *integration* by name. Names are only unique per
                ``(account, type)``, so this is required when *integration* is
                a name; it is ignored when *integration* is an ID.
            space: Optional space ID or name. This is only a visibility
                filter, not required to resolve a name.

        Returns:
            This method returns None on success (common empty 204 response).

        Raises:
            NotFoundError: If *integration* is a name and *integration_type*
                is not provided, or the name cannot be resolved.
            ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        integration_id = _find_integration_id(
            api=self._api,
            integration=integration,
            integration_type=integration_type,
            space=space,
        )
        self._api.delete_integration(integration_id=integration_id)
