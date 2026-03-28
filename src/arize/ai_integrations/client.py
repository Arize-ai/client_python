"""Client implementation for managing AI integrations in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import (
    find_ai_integration_id,
    resolve_resource,
)

if TYPE_CHECKING:
    # builtins is needed to use builtins.list in type annotations because
    # the class has a list() method that shadows the built-in list type
    import builtins

    from arize._generated.api_client import models
    from arize._generated.api_client.api_client import ApiClient
    from arize._generated.api_client.models.ai_integration_auth_type import (
        AiIntegrationAuthType,
    )
    from arize._generated.api_client.models.ai_integration_provider import (
        AiIntegrationProvider,
    )
    from arize._generated.api_client.models.ai_integration_scoping import (
        AiIntegrationScoping,
    )
    from arize.config import SDKConfiguration

logger = logging.getLogger(__name__)

# Sentinel object used to distinguish "caller did not pass this argument" from
# "caller explicitly passed None" in update().  This matters because the
# generated pydantic model uses ``model_fields_set`` to decide whether to
# serialize a nullable field as JSON ``null`` (clearing it on the server) vs.
# omitting it entirely (leaving it unchanged).
_UNSET: Any = object()


class AiIntegrationsClient:
    """Client for managing Arize AI integrations.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The AI integrations client is a thin wrapper around the generated REST API client,
    using the shared generated API client owned by
    :class:`arize.config.SDKConfiguration`.

    AI integrations configure access to external LLM providers (OpenAI, Azure OpenAI,
    AWS Bedrock, Vertex AI, Anthropic, and others) for use within the Arize platform.
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

        self._api = gen.AIIntegrationsApi(generated_client)

    @prerelease_endpoint(key="ai_integrations.list", stage=ReleaseStage.ALPHA)
    def list(
        self,
        *,
        name: str | None = None,
        space: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> models.AiIntegrationsList200Response:
        """List AI integrations the user has access to.

        This endpoint supports cursor-based pagination. When provided,
        ``space`` filters results to a particular space.

        Args:
            name: Optional case-insensitive substring filter on the integration name.
            space: Optional space filter. If the value is a base64-encoded resource ID it is
                treated as a space ID; otherwise it is used as a case-insensitive
                substring filter on the space name.
            limit: Maximum number of integrations to return. The server may enforce
                an upper bound (max 100).
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated AI integrations list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails.
        """
        resolved_space = resolve_resource(space)
        return self._api.ai_integrations_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=name,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="ai_integrations.get", stage=ReleaseStage.ALPHA)
    def get(
        self, *, integration: str, space: str | None = None
    ) -> models.AiIntegration:
        """Get an AI integration by name or ID.

        Args:
            integration: Integration name or ID to retrieve.
            space: Optional space name or ID used to disambiguate when
                resolving by name.

        Returns:
            The AI integration object.

        Raises:
            ApiException: If the API request fails
                (for example, integration not found).
        """
        integration_id = find_ai_integration_id(
            api=self._api,
            integration=integration,
            space=space,
        )
        return self._api.ai_integrations_get(integration_id=integration_id)

    @prerelease_endpoint(key="ai_integrations.create", stage=ReleaseStage.ALPHA)
    def create(
        self,
        *,
        name: str,
        provider: AiIntegrationProvider,
        api_key: str | None = None,
        base_url: str | None = None,
        model_names: builtins.list[str] | None = None,
        headers: dict[str, str] | None = None,
        enable_default_models: bool | None = None,
        function_calling_enabled: bool | None = None,
        auth_type: AiIntegrationAuthType | None = None,
        provider_metadata: dict[str, Any] | None = None,
        scopings: builtins.list[AiIntegrationScoping] | None = None,
    ) -> models.AiIntegration:
        """Create a new AI integration.

        Integration names must be unique within the account.

        For ``awsBedrock`` provider, ``provider_metadata`` must include ``role_arn``.
        For ``vertexAI`` provider, ``provider_metadata`` must include ``project_id``,
        ``location``, and ``project_access_label``.

        Args:
            name: Integration name (must be unique within the account).
            provider: LLM provider (e.g. ``openAI``, ``azureOpenAI``, ``awsBedrock``,
                ``vertexAI``, ``anthropic``, ``custom``).
            api_key: API key for the provider (write-only, never returned).
            base_url: Custom base URL for the provider.
            model_names: Supported model names.
            headers: Custom headers to include in requests.
            enable_default_models: Enable the provider's default model list.
                Defaults to ``False`` if not provided.
            function_calling_enabled: Enable function/tool calling.
                Defaults to ``True`` if not provided.
            auth_type: Authentication type. Defaults to ``default`` if not provided.
            provider_metadata: Provider-specific configuration (AWS or GCP metadata).
            scopings: Visibility scoping rules. Defaults to account-wide if omitted.

        Returns:
            The created AI integration object.

        Raises:
            ApiException: If the API request fails.
        """
        from arize._generated import api_client as gen

        body = gen.AiIntegrationsCreateRequest(
            name=name,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model_names=model_names,
            headers=headers,
            enable_default_models=enable_default_models,
            function_calling_enabled=function_calling_enabled,
            auth_type=auth_type,
            provider_metadata=provider_metadata,
            scopings=scopings,
        )
        return self._api.ai_integrations_create(
            ai_integrations_create_request=body
        )

    @prerelease_endpoint(key="ai_integrations.update", stage=ReleaseStage.ALPHA)
    def update(
        self,
        *,
        integration: str,
        space: str | None = None,
        name: str | None = _UNSET,
        provider: AiIntegrationProvider | None = _UNSET,
        api_key: str | None = _UNSET,
        base_url: str | None = _UNSET,
        model_names: builtins.list[str] | None = _UNSET,
        headers: dict[str, str] | None = _UNSET,
        enable_default_models: bool | None = _UNSET,
        function_calling_enabled: bool | None = _UNSET,
        auth_type: AiIntegrationAuthType | None = _UNSET,
        provider_metadata: dict[str, Any] | None = _UNSET,
        scopings: builtins.list[AiIntegrationScoping] | None = _UNSET,
    ) -> models.AiIntegration:
        """Update an AI integration by name or ID.

        Only the fields you pass are sent to the server. Omitted fields are
        left unchanged. To explicitly clear a nullable field (e.g.
        ``api_key``), pass ``None``.

        Args:
            integration: Integration name or ID to update.
            space: Optional space name or ID used to disambiguate when
                resolving by name.
            name: Updated integration name.
            provider: Updated LLM provider.
            api_key: New API key. Pass ``None`` to clear the existing key.
            base_url: Updated custom base URL. Pass ``None`` to clear.
            model_names: Updated model names (replaces all existing).
            headers: Updated custom headers. Pass ``None`` to clear.
            enable_default_models: Updated default models flag.
            function_calling_enabled: Updated function calling flag.
            auth_type: Updated authentication type.
            provider_metadata: Updated provider-specific configuration.
                Pass ``None`` to clear.
            scopings: Updated visibility scoping rules (replaces all existing).

        Returns:
            The updated AI integration object.

        Raises:
            ApiException: If the API request fails
                (for example, integration not found or insufficient permissions).
        """
        from arize._generated import api_client as gen

        # Build kwargs with only the fields the caller actually provided so
        # that pydantic's model_fields_set accurately reflects intent.  This
        # prevents nullable fields (api_key, base_url, headers,
        # provider_metadata) from being serialized as JSON null when the
        # caller didn't mention them.
        kwargs: dict[str, Any] = {
            k: v
            for k, v in (
                ("name", name),
                ("provider", provider),
                ("api_key", api_key),
                ("base_url", base_url),
                ("model_names", model_names),
                ("headers", headers),
                ("enable_default_models", enable_default_models),
                ("function_calling_enabled", function_calling_enabled),
                ("auth_type", auth_type),
                ("provider_metadata", provider_metadata),
                ("scopings", scopings),
            )
            if v is not _UNSET
        }

        integration_id = find_ai_integration_id(
            api=self._api,
            integration=integration,
            space=space,
        )

        body = gen.AiIntegrationsUpdateRequest(**kwargs)
        return self._api.ai_integrations_update(
            integration_id=integration_id,
            ai_integrations_update_request=body,
        )

    @prerelease_endpoint(key="ai_integrations.delete", stage=ReleaseStage.ALPHA)
    def delete(self, *, integration: str, space: str | None = None) -> None:
        """Delete an AI integration by name or ID.

        This operation is irreversible.

        Args:
            integration: Integration name or ID to delete.
            space: Optional space name or ID used to disambiguate when
                resolving by name.

        Raises:
            ApiException: If the API request fails
                (for example, integration not found or insufficient permissions).
        """
        integration_id = find_ai_integration_id(
            api=self._api,
            integration=integration,
            space=space,
        )
        self._api.ai_integrations_delete(integration_id=integration_id)
