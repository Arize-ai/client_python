"""Client implementation for managing prompts in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import (
    find_prompt_id,
    find_space_id,
    resolve_resource,
)

if TYPE_CHECKING:
    import builtins

    from arize._generated.api_client import models
    from arize._generated.api_client.api_client import ApiClient
    from arize._generated.api_client.models.input_variable_format import (
        InputVariableFormat,
    )
    from arize._generated.api_client.models.invocation_params import (
        InvocationParams,
    )
    from arize._generated.api_client.models.llm_message import LLMMessage
    from arize._generated.api_client.models.llm_provider import LlmProvider
    from arize._generated.api_client.models.provider_params import (
        ProviderParams,
    )
    from arize.config import SDKConfiguration

logger = logging.getLogger(__name__)


class PromptsClient:
    """Client for managing prompts in the Arize platform.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The prompts client is a thin wrapper around the generated REST API client,
    using the shared generated API client owned by
    :class:`arize.config.SDKConfiguration`.
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

        # Import at runtime so it's still lazy and extras-gated by the parent
        from arize._generated import api_client as gen

        # Use the provided client directly
        self._api = gen.PromptsApi(generated_client)
        self._spaces_api = gen.SpacesApi(generated_client)

    @prerelease_endpoint(key="prompts.list", stage=ReleaseStage.ALPHA)
    def list(
        self,
        *,
        name: str | None = None,
        space: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> models.PromptsList200Response:
        """List prompts in a space.

        Args:
            name: Optional case-insensitive substring filter on the prompt name.
            space: Optional space filter. If the value is a base64-encoded resource ID it is
                treated as a space ID; otherwise it is used as a case-insensitive
                substring filter on the space name.
            limit: Maximum number of prompts to return. The server enforces an
                upper bound of 100.
            cursor: Opaque pagination cursor returned from a previous response.

        Returns:
            A response object with the prompts and pagination information.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 401/403/429).
        """
        resolved_space = resolve_resource(space)
        return self._api.prompts_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=name,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="prompts.create", stage=ReleaseStage.ALPHA)
    def create(
        self,
        *,
        space: str,
        name: str,
        commit_message: str,
        input_variable_format: InputVariableFormat,
        provider: LlmProvider,
        messages: builtins.list[LLMMessage],
        description: str | None = None,
        model: str | None = None,
        invocation_params: InvocationParams | None = None,
        provider_params: ProviderParams | None = None,
    ) -> models.PromptWithVersion:
        """Create a prompt with an initial version.

        Args:
            space: Space ID or name to create the prompt in. If a name is
                provided it will be resolved to a space ID automatically.
            name: Prompt name (must be unique within the space).
            commit_message: Commit message describing the initial version.
            input_variable_format: Variable interpolation format for the prompt
                template (e.g. ``InputVariableFormat.F_STRING``).
            provider: LLM provider for the prompt.
            messages: Messages that make up the prompt template (at least one required).
            description: Optional description of the prompt.
            model: Optional model name. If omitted, no default model is set on the
                version.
            invocation_params: Optional invocation parameters (e.g. temperature,
                max_tokens).
            provider_params: Optional provider-specific parameters.

        Returns:
            The created prompt with its initial version.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/409/429).
        """
        space_id = find_space_id(self._spaces_api, space)

        from arize._generated import api_client as gen

        version = gen.PromptVersionCreateRequest(
            commit_message=commit_message,
            input_variable_format=input_variable_format,
            provider=provider,
            model=model,
            messages=messages,
            invocation_params=invocation_params,
            provider_params=provider_params,
        )
        body = gen.PromptsCreateRequest(
            space_id=space_id,
            name=name,
            description=description,
            version=version,
        )
        return self._api.prompts_create(prompts_create_request=body)

    @prerelease_endpoint(key="prompts.get", stage=ReleaseStage.ALPHA)
    def get(
        self,
        *,
        prompt: str,
        space: str | None = None,
        version_id: str | None = None,
        label: str | None = None,
    ) -> models.PromptWithVersion:
        """Get a prompt by ID or name.

        Optionally resolves a specific version by ``version_id`` or a ``label``.
        If neither is supplied, the latest version is returned.

        Args:
            prompt: Prompt ID or name. If a name is provided, ``space`` must
                also be supplied so the name can be resolved.
            space: Optional space ID or name. Required when *prompt* is a name.
            version_id: Optional specific version ID to retrieve.
            label: Optional label name to resolve to a version (e.g. ``"production"``).

        Returns:
            The prompt object with its resolved version.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        prompt_id = find_prompt_id(
            api=self._api,
            prompt=prompt,
            space=space,
        )
        return self._api.prompts_get(
            prompt_id=prompt_id,
            version_id=version_id,
            label=label,
        )

    @prerelease_endpoint(key="prompts.update", stage=ReleaseStage.ALPHA)
    def update(
        self,
        *,
        prompt: str,
        space: str | None = None,
        description: str,
    ) -> models.Prompt:
        """Update a prompt's metadata.

        Args:
            prompt: Prompt ID or name. If a name is provided, ``space`` must
                also be supplied so the name can be resolved.
            space: Optional space ID or name. Required when *prompt* is a name.
            description: Updated description for the prompt.

        Returns:
            The updated prompt object.

        Raises:
            ValueError: If no fields to update are provided.
            ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        prompt_id = find_prompt_id(
            api=self._api,
            prompt=prompt,
            space=space,
        )

        from arize._generated import api_client as gen

        body = gen.PromptsUpdateRequest(description=description)
        return self._api.prompts_update(
            prompt_id=prompt_id, prompts_update_request=body
        )

    @prerelease_endpoint(key="prompts.delete", stage=ReleaseStage.ALPHA)
    def delete(self, *, prompt: str, space: str | None = None) -> None:
        """Delete a prompt by ID or name.

        This operation is irreversible and removes all associated versions.

        Args:
            prompt: Prompt ID or name. If a name is provided, ``space`` must
                also be supplied so the name can be resolved.
            space: Optional space ID or name. Required when *prompt* is a name.

        Returns:
            None on success (204 No Content).

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        prompt_id = find_prompt_id(
            api=self._api,
            prompt=prompt,
            space=space,
        )
        return self._api.prompts_delete(prompt_id=prompt_id)

    @prerelease_endpoint(key="prompts.list_versions", stage=ReleaseStage.ALPHA)
    def list_versions(
        self,
        *,
        prompt: str,
        space: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> models.PromptVersionsList200Response:
        """List versions for a prompt.

        Args:
            prompt: Prompt ID or name. If a name is provided, ``space`` must
                also be supplied so the name can be resolved.
            space: Optional space ID or name. Required when *prompt* is a name.
            limit: Maximum number of versions to return. The server enforces an
                upper bound of 100.
            cursor: Opaque pagination cursor returned from a previous response.

        Returns:
            A response object with the prompt versions and pagination information.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        prompt_id = find_prompt_id(
            api=self._api,
            prompt=prompt,
            space=space,
        )
        return self._api.prompt_versions_list(
            prompt_id=prompt_id,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="prompts.create_version", stage=ReleaseStage.ALPHA)
    def create_version(
        self,
        *,
        prompt: str,
        space: str | None = None,
        commit_message: str,
        input_variable_format: InputVariableFormat,
        provider: LlmProvider,
        messages: builtins.list[LLMMessage],
        model: str | None = None,
        invocation_params: InvocationParams | None = None,
        provider_params: ProviderParams | None = None,
    ) -> models.PromptVersion:
        """Create a new version for an existing prompt.

        Args:
            prompt: Prompt ID or name. If a name is provided, ``space`` must
                also be supplied so the name can be resolved.
            space: Optional space ID or name. Required when *prompt* is a name.
            commit_message: Commit message describing this version.
            input_variable_format: Variable interpolation format for the prompt
                template (e.g. ``InputVariableFormat.F_STRING``).
            provider: LLM provider for this version.
            messages: Messages that make up the prompt template (at least one required).
            model: Optional model name. If omitted, no default model is set on this
                version.
            invocation_params: Optional invocation parameters (e.g. temperature,
                max_tokens).
            provider_params: Optional provider-specific parameters.

        Returns:
            The created prompt version.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/404/429).
        """
        prompt_id = find_prompt_id(
            api=self._api,
            prompt=prompt,
            space=space,
        )

        from arize._generated import api_client as gen

        body = gen.PromptVersionsCreateRequest(
            commit_message=commit_message,
            input_variable_format=input_variable_format,
            provider=provider,
            model=model,
            messages=messages,
            invocation_params=invocation_params,
            provider_params=provider_params,
        )
        return self._api.prompt_versions_create(
            prompt_id=prompt_id, prompt_versions_create_request=body
        )

    @prerelease_endpoint(key="prompts.get_label", stage=ReleaseStage.ALPHA)
    def get_label(
        self, *, prompt: str, space: str | None = None, label_name: str
    ) -> models.PromptVersion:
        """Resolve a label to a prompt version.

        Args:
            prompt: Prompt ID or name. If a name is provided, ``space`` must
                also be supplied so the name can be resolved.
            space: Optional space ID or name. Required when *prompt* is a name.
            label_name: Label name to resolve (e.g. ``"production"``, ``"staging"``).

        Returns:
            The prompt version the label currently points to.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        prompt_id = find_prompt_id(
            api=self._api,
            prompt=prompt,
            space=space,
        )
        return self._api.prompt_labels_get(
            prompt_id=prompt_id, label_name=label_name
        )

    @prerelease_endpoint(key="prompts.set_labels", stage=ReleaseStage.ALPHA)
    def set_labels(
        self,
        *,
        version_id: str,
        labels: builtins.list[str],
    ) -> models.PromptVersionLabelsSet200Response:
        """Set labels on a prompt version.

        Replaces all existing labels on the version with the provided list.

        Args:
            version_id: Version ID to set labels on.
            labels: List of label names to assign (replaces all existing labels).

        Returns:
            The response with the updated labels.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/404/429).
        """
        from arize._generated import api_client as gen

        body = gen.PromptVersionLabelsSetRequest(labels=labels)
        return self._api.prompt_version_labels_set(
            version_id=version_id, prompt_version_labels_set_request=body
        )

    @prerelease_endpoint(key="prompts.delete_label", stage=ReleaseStage.ALPHA)
    def delete_label(self, *, version_id: str, label_name: str) -> None:
        """Remove a label from a prompt version.

        Args:
            version_id: Version ID to remove the label from.
            label_name: Label name to remove (e.g. ``"production"``, ``"staging"``).

        Returns:
            None on success (204 No Content).

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        return self._api.prompt_version_labels_delete(
            version_id=version_id, label_name=label_name
        )
