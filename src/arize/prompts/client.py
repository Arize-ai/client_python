"""Client implementation for managing prompts in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.constants.config import DEFAULT_LIST_LIMIT
from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.prompts.types import PromptVersion, PromptWithVersion
from arize.utils.resolve import (
    _find_prompt_id,
    _find_space_id,
    _resolve_resource,
)

if TYPE_CHECKING:
    import builtins

    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration
    from arize.prompts.types import (
        InputVariableFormat,
        InvocationParams,
        ListPromptsResponse,
        ListPromptVersionsResponse,
        LLMMessage,
        LlmProvider,
        Prompt,
        ProviderParams,
    )

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

    @prerelease_endpoint(key="prompts.list", stage=ReleaseStage.BETA)
    def list(
        self,
        *,
        name: str | None = None,
        space: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> ListPromptsResponse:
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
        resolved_space = _resolve_resource(space)
        return self._api.list_prompts(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=name,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="prompts.create", stage=ReleaseStage.BETA)
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
    ) -> PromptWithVersion:
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
        space_id = _find_space_id(self._spaces_api, space)

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
        body = gen.CreatePromptRequest(
            space_id=space_id,
            name=name,
            description=description,
            version=version,
        )
        result = self._api.create_prompt(create_prompt_request=body)
        return PromptWithVersion.model_validate(result, from_attributes=True)

    @prerelease_endpoint(key="prompts.get", stage=ReleaseStage.BETA)
    def get(
        self,
        *,
        prompt: str,
        space: str | None = None,
        version_id: str | None = None,
        label: str | None = None,
    ) -> PromptWithVersion:
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
        prompt_id = _find_prompt_id(
            api=self._api,
            prompt=prompt,
            space=space,
        )
        result = self._api.get_prompt(
            prompt_id=prompt_id,
            version_id=version_id,
            label=label,
        )
        return PromptWithVersion.model_validate(result, from_attributes=True)

    @prerelease_endpoint(key="prompts.get_version", stage=ReleaseStage.BETA)
    def get_version(self, *, version_id: str) -> PromptVersion:
        """Get a single prompt version by its ID.

        Version IDs are pure IDs with no name resolution.

        Args:
            version_id: Version ID to retrieve.

        Returns:
            The prompt version.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/404/429).
        """
        result = self._api.get_prompt_version(version_id=version_id)
        return PromptVersion.model_validate(result, from_attributes=True)

    @prerelease_endpoint(key="prompts.update", stage=ReleaseStage.BETA)
    def update(
        self,
        *,
        prompt: str,
        space: str | None = None,
        description: str,
    ) -> Prompt:
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
        prompt_id = _find_prompt_id(
            api=self._api,
            prompt=prompt,
            space=space,
        )

        from arize._generated import api_client as gen

        body = gen.UpdatePromptRequest(description=description)
        return self._api.update_prompt(
            prompt_id=prompt_id, update_prompt_request=body
        )

    @prerelease_endpoint(key="prompts.delete", stage=ReleaseStage.BETA)
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
        prompt_id = _find_prompt_id(
            api=self._api,
            prompt=prompt,
            space=space,
        )
        return self._api.delete_prompt(prompt_id=prompt_id)

    @prerelease_endpoint(key="prompts.list_versions", stage=ReleaseStage.BETA)
    def list_versions(
        self,
        *,
        prompt: str,
        space: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> ListPromptVersionsResponse:
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
        prompt_id = _find_prompt_id(
            api=self._api,
            prompt=prompt,
            space=space,
        )
        return self._api.list_prompt_versions(
            prompt_id=prompt_id,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="prompts.create_version", stage=ReleaseStage.BETA)
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
    ) -> PromptVersion:
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
        prompt_id = _find_prompt_id(
            api=self._api,
            prompt=prompt,
            space=space,
        )

        from arize._generated import api_client as gen

        body = gen.CreatePromptVersionRequest(
            commit_message=commit_message,
            input_variable_format=input_variable_format,
            provider=provider,
            model=model,
            messages=messages,
            invocation_params=invocation_params,
            provider_params=provider_params,
        )
        result = self._api.create_prompt_version(
            prompt_id=prompt_id, create_prompt_version_request=body
        )
        return PromptVersion.model_validate(result, from_attributes=True)

    @prerelease_endpoint(
        key="prompts.get_version_by_label", stage=ReleaseStage.BETA
    )
    def get_version_by_label(
        self, *, prompt: str, space: str | None = None, label_name: str
    ) -> PromptVersion:
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
        prompt_id = _find_prompt_id(
            api=self._api,
            prompt=prompt,
            space=space,
        )
        result = self._api.get_prompt_label(
            prompt_id=prompt_id, label_name=label_name
        )
        return PromptVersion.model_validate(result, from_attributes=True)

    @prerelease_endpoint(key="prompts.set_labels", stage=ReleaseStage.BETA)
    def set_labels(
        self,
        *,
        version_id: str,
        labels: builtins.list[str],
    ) -> PromptVersion:
        """Set labels on a prompt version.

        Replaces all existing labels on the version with the provided list.

        Args:
            version_id: Version ID to set labels on.
            labels: List of label names to assign (replaces all existing labels).

        Returns:
            The updated prompt version.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/404/429).
        """
        from arize._generated import api_client as gen

        body = gen.SetPromptVersionLabelsRequest(labels=labels)
        result = self._api.set_prompt_version_label(
            version_id=version_id, set_prompt_version_labels_request=body
        )
        return PromptVersion.model_validate(result, from_attributes=True)

    @prerelease_endpoint(key="prompts.delete_label", stage=ReleaseStage.BETA)
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
        return self._api.delete_prompt_version_label(
            version_id=version_id, label_name=label_name
        )
