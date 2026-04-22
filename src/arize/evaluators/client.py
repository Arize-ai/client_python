"""Client implementation for managing evaluators in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import (
    _find_evaluator_id,
    _find_space_id,
    _resolve_resource,
)

if TYPE_CHECKING:
    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration
    from arize.evaluators.types import (
        Evaluator,
        EvaluatorsList200Response,
        EvaluatorVersion,
        EvaluatorVersionsList200Response,
        EvaluatorWithVersion,
        TemplateConfig,
    )


logger = logging.getLogger(__name__)


class EvaluatorsClient:
    """Client for managing Arize evaluators and evaluator versions.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The evaluators client is a thin wrapper around the generated REST API client,
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

        self._api = gen.EvaluatorsApi(generated_client)
        self._spaces_api = gen.SpacesApi(generated_client)

    # -------------------------------------------------------------------------
    # Evaluators
    # -------------------------------------------------------------------------

    @prerelease_endpoint(key="evaluators.list", stage=ReleaseStage.ALPHA)
    def list(
        self,
        *,
        name: str | None = None,
        space: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> EvaluatorsList200Response:
        """List evaluators the user has access to.

        Results are sorted by update date (most recent first). This endpoint
        supports cursor-based pagination. When ``space`` is provided, results
        are limited to that space; otherwise evaluators from all permitted spaces
        are returned.

        Args:
            name: Optional case-insensitive substring filter on the evaluator name.
            space: Optional space filter. If the value is a base64-encoded resource ID it is
                treated as a space ID; otherwise it is used as a case-insensitive
                substring filter on the space name.
            limit: Maximum number of evaluators to return (1-100).
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated evaluator list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails.
        """
        resolved_space = _resolve_resource(space)
        return self._api.evaluators_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=name,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="evaluators.get", stage=ReleaseStage.ALPHA)
    def get(
        self,
        *,
        evaluator: str,
        space: str | None = None,
        version_id: str | None = None,
    ) -> EvaluatorWithVersion:
        """Get an evaluator by name or ID, with its resolved version.

        By default, the latest version is returned. Pass ``version_id`` to
        resolve a specific version instead.

        Args:
            evaluator: Evaluator name or global ID (base64) to retrieve.
            space: Optional space name or ID. Required when ``evaluator`` is a
                name rather than an ID.
            version_id: Optional version global ID (base64). If omitted, the
                latest version is returned.

        Returns:
            The evaluator with its resolved version.

        Raises:
            ApiException: If the API request fails
                (for example, evaluator not found).
        """
        evaluator_id = _find_evaluator_id(
            api=self._api,
            evaluator=evaluator,
            space=space,
        )
        return self._api.evaluators_get(
            evaluator_id=evaluator_id,
            version_id=version_id,
        )

    @prerelease_endpoint(key="evaluators.create", stage=ReleaseStage.ALPHA)
    def create(
        self,
        *,
        name: str,
        space: str,
        evaluator_type: Literal["template", "code"] = "template",
        commit_message: str,
        template_config: TemplateConfig,
        description: str | None = None,
    ) -> EvaluatorWithVersion:
        r"""Create a new evaluator with an initial version.

        The evaluator ``name`` must be unique within the given space.

        Currently, only ``"template"`` evaluators are supported. The ``evaluator_type``
        parameter is accepted for forward-compatibility but raises
        ``ValueError`` if set to anything other than ``"template"``.

        Args:
            name: Evaluator name (must be unique within the space).
            space: Space name or ID to create the evaluator in.
            evaluator_type: Evaluator type. Only ``"template"`` is supported;
                the parameter is accepted for forward-compatibility.
            commit_message: Commit message for the initial version.
            template_config: Template configuration for the initial version.
                Build this with :class:`arize.evaluators.types.TemplateConfig`.
                Required fields:

                - ``name`` — eval column name; must match
                  ``^[a-zA-Z0-9_\\s\\-&()]+$``.
                - ``template`` — prompt template string with ``{{variable}}``
                  placeholders referencing span/trace attributes.
                - ``include_explanations`` — whether the LLM should include a
                  reasoning explanation alongside the score.
                - ``use_function_calling_if_available`` — prefer structured
                  function-call output over free-text parsing when the model
                  supports it.
                - ``llm_config`` — :class:`arize.evaluators.types.EvaluatorLlmConfig`
                  specifying the model provider, model name, and API key.

                Optional fields:

                - ``classification_choices`` — ``dict[str, float]`` mapping
                  label → numeric score (e.g. ``{"relevant": 1,
                  "irrelevant": 0}``). When omitted the evaluator produces
                  freeform (non-classification) output.
                - ``direction`` — ``"maximize"`` or ``"minimize"``, the
                  optimization direction for annotation scores.
                - ``data_granularity`` — ``"span"``, ``"trace"``, or
                  ``"session"``.
            description: Optional human-readable description of the evaluator.

        Returns:
            The created evaluator with its initial version.

        Raises:
            ApiException: If the API request fails
                (for example, name conflict or invalid payload).
        """
        from arize._generated import api_client as gen

        if evaluator_type != "template":
            raise ValueError(
                f"Evaluator type {evaluator_type} is not supported"
            )

        space_id = _find_space_id(self._spaces_api, space)

        version = gen.EvaluatorsCreateRequestVersion(
            commit_message=commit_message,
            template_config=template_config,
        )
        body = gen.EvaluatorsCreateRequest(
            name=name,
            space_id=space_id,
            type=evaluator_type,
            description=description,
            version=version,
        )
        return self._api.evaluators_create(evaluators_create_request=body)

    @prerelease_endpoint(key="evaluators.update", stage=ReleaseStage.ALPHA)
    def update(
        self,
        *,
        evaluator: str,
        space: str | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> Evaluator:
        """Update an evaluator's metadata.

        Args:
            evaluator: Evaluator name or global ID (base64) to update.
            space: Optional space name or ID. Required when ``evaluator`` is a
                name rather than an ID.
            name: New evaluator name (must be unique within its space).
            description: New description for the evaluator.

        Returns:
            The updated evaluator.

        Raises:
            ApiException: If the API request fails.
        """
        evaluator_id = _find_evaluator_id(
            api=self._api,
            evaluator=evaluator,
            space=space,
        )

        from arize._generated import api_client as gen

        body = gen.EvaluatorsUpdateRequest(name=name, description=description)
        return self._api.evaluators_update(
            evaluator_id=evaluator_id,
            evaluators_update_request=body,
        )

    @prerelease_endpoint(key="evaluators.delete", stage=ReleaseStage.ALPHA)
    def delete(self, *, evaluator: str, space: str | None = None) -> None:
        """Delete an evaluator and all its versions.

        This operation is irreversible.

        Args:
            evaluator: Evaluator name or global ID (base64) to delete.
            space: Optional space name or ID. Required when ``evaluator`` is a
                name rather than an ID.

        Returns:
            None.

        Raises:
            ApiException: If the API request fails
                (for example, evaluator not found).
        """
        evaluator_id = _find_evaluator_id(
            api=self._api,
            evaluator=evaluator,
            space=space,
        )
        self._api.evaluators_delete(evaluator_id=evaluator_id)

    # -------------------------------------------------------------------------
    # Evaluator versions
    # -------------------------------------------------------------------------

    @prerelease_endpoint(
        key="evaluators.list_versions", stage=ReleaseStage.ALPHA
    )
    def list_versions(
        self,
        *,
        evaluator: str,
        space: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> EvaluatorVersionsList200Response:
        """List all versions of an evaluator.

        Results are returned with cursor-based pagination.

        Args:
            evaluator: Evaluator name or global ID (base64) to list versions for.
            space: Optional space name or ID. Required when ``evaluator`` is a
                name rather than an ID.
            limit: Maximum number of versions to return (1-100).
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated evaluator version list response.

        Raises:
            ApiException: If the API request fails.
        """
        evaluator_id = _find_evaluator_id(
            api=self._api,
            evaluator=evaluator,
            space=space,
        )
        return self._api.evaluator_versions_list(
            evaluator_id=evaluator_id,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="evaluators.get_version", stage=ReleaseStage.ALPHA)
    def get_version(self, *, version_id: str) -> EvaluatorVersion:
        """Get a specific evaluator version by its global ID.

        Args:
            version_id: Evaluator version global ID (base64).

        Returns:
            The evaluator version.

        Raises:
            ApiException: If the API request fails
                (for example, version not found).
        """
        return self._api.evaluator_versions_get(version_id=version_id)

    @prerelease_endpoint(
        key="evaluators.create_version", stage=ReleaseStage.ALPHA
    )
    def create_version(
        self,
        *,
        evaluator: str,
        space: str | None = None,
        commit_message: str,
        template_config: TemplateConfig,
    ) -> EvaluatorVersion:
        r"""Create a new version of an existing evaluator.

        The new version becomes the latest version immediately (versioning is
        append-only).

        Versions are immutable once created. To change the configuration, create a new version.

        Args:
            evaluator: Evaluator name or global ID (base64) to add a version to.
            space: Optional space name or ID. Required when ``evaluator`` is a
                name rather than an ID.
            commit_message: Commit message describing the changes in this version.
            template_config: Updated template configuration for this version.
                Build this with :class:`arize.evaluators.types.TemplateConfig`.
                Required fields:

                - ``name`` — eval column name; must match
                  ``^[a-zA-Z0-9_\\s\\-&()]+$``.
                - ``template`` — prompt template string with ``{{variable}}``
                  placeholders referencing span/trace attributes.
                - ``include_explanations`` — whether the LLM should include a
                  reasoning explanation alongside the score.
                - ``use_function_calling_if_available`` — prefer structured
                  function-call output over free-text parsing when the model
                  supports it.
                - ``llm_config`` — :class:`arize.evaluators.types.EvaluatorLlmConfig`
                  specifying the model provider, model name, and API key.

                Optional fields:

                - ``classification_choices`` — ``dict[str, float]`` mapping
                  label → numeric score (e.g. ``{"relevant": 1,
                  "irrelevant": 0}``). When omitted the evaluator produces
                  freeform (non-classification) output.
                - ``direction`` — ``"maximize"`` or ``"minimize"``, the
                  optimization direction for annotation scores.
                - ``data_granularity`` — ``"span"``, ``"trace"``, or
                  ``"session"``.

        Returns:
            The newly created evaluator version.

        Raises:
            ApiException: If the API request fails.
        """
        evaluator_id = _find_evaluator_id(
            api=self._api,
            evaluator=evaluator,
            space=space,
        )

        from arize._generated import api_client as gen

        body = gen.EvaluatorVersionsCreateRequest(
            commit_message=commit_message,
            template_config=template_config,
        )
        return self._api.evaluator_versions_create(
            evaluator_id=evaluator_id,
            evaluator_versions_create_request=body,
        )
