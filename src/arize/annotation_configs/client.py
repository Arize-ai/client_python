"""Client implementation for managing annotation configs in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize._utils import unwrap_oneof
from arize.annotation_configs.types import (
    AnnotationConfigType,
    ListAnnotationConfigsResponse,
)
from arize.constants.config import DEFAULT_LIST_LIMIT
from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import (
    _find_annotation_config_id,
    _find_space_id,
    _resolve_resource,
)

if TYPE_CHECKING:
    import builtins

    from arize._generated.api_client.api_client import ApiClient
    from arize.annotation_configs.types import (
        CategoricalAnnotationConfig,
        CategoricalAnnotationValue,
        ContinuousAnnotationConfig,
        FreeformAnnotationConfig,
        OptimizationDirection,
    )
    from arize.config import SDKConfiguration

logger = logging.getLogger(__name__)


class AnnotationConfigsClient:
    """Client for managing annotation configs including creation, retrieval, and deletion.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The annotation configs client is a thin wrapper around the generated REST API client,
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
        self._api = gen.AnnotationConfigsApi(generated_client)
        self._spaces_api = gen.SpacesApi(generated_client)

    @prerelease_endpoint(key="annotation_configs.list", stage=ReleaseStage.BETA)
    def list(
        self,
        *,
        name: str | None = None,
        space: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> ListAnnotationConfigsResponse:
        """List annotation configs the user has access to.

        Annotation configs are returned in descending creation order (most recently created
        first).

        Args:
            name: Optional case-insensitive substring filter on the annotation config name.
            space: Optional space filter. If the value is a base64-encoded resource ID it is
                treated as a space ID; otherwise it is used as a case-insensitive
                substring filter on the space name.
            limit: Maximum number of annotation configs to return. The server enforces an
                upper bound.
            cursor: Opaque pagination cursor returned from a previous response.

        Returns:
            A response object with the annotation configs and pagination information.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 401/403/429).
        """
        resolved_space = _resolve_resource(space)
        result = self._api.list_annotation_configs(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=name,
            limit=limit,
            cursor=cursor,
        )
        return ListAnnotationConfigsResponse.model_validate(
            result, from_attributes=True
        )

    @prerelease_endpoint(
        key="annotation_configs.create", stage=ReleaseStage.BETA
    )
    def create_continuous(
        self,
        *,
        name: str,
        space: str,
        minimum_score: float | int,
        maximum_score: float | int,
        optimization_direction: OptimizationDirection | None = None,
    ) -> ContinuousAnnotationConfig:
        """Create a continuous annotation config.

        Continuous annotation configs let a scorer enter a numeric score
        within a fixed range, e.g. a 0-1 quality score.

        Args:
            name: Annotation config name (must be unique within the target space).
            space: Space ID or name to create the annotation config in.
            minimum_score: Minimum score a scorer is allowed to submit.
            maximum_score: Maximum score a scorer is allowed to submit.
            optimization_direction: Optional direction (e.g. maximize or
                minimize) that indicates which end of the score range is
                considered better.

        Returns:
            The created continuous annotation config.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/409/429).
        """
        from arize._generated import api_client as gen

        space_id = _find_space_id(self._spaces_api, space)
        body = gen.CreateAnnotationConfigRequest(
            actual_instance=gen.CreateContinuousAnnotationConfigRequest(
                name=name,
                space_id=space_id,
                annotation_config_type=AnnotationConfigType.CONTINUOUS.value,
                minimum_score=minimum_score,
                maximum_score=maximum_score,
                optimization_direction=optimization_direction,
            )
        )
        result = self._api.create_annotation_config(
            create_annotation_config_request=body
        )
        return unwrap_oneof(result)  # type: ignore[return-value]

    @prerelease_endpoint(
        key="annotation_configs.create", stage=ReleaseStage.BETA
    )
    def create_categorical(
        self,
        *,
        name: str,
        space: str,
        values: builtins.list[CategoricalAnnotationValue],
        optimization_direction: OptimizationDirection | None = None,
    ) -> CategoricalAnnotationConfig:
        """Create a categorical annotation config.

        Categorical annotation configs let a scorer choose from a fixed
        set of labeled values, e.g. "correct" / "incorrect".

        Args:
            name: Annotation config name (must be unique within the target space).
            space: Space ID or name to create the annotation config in.
            values: The labeled values a scorer can choose from.
            optimization_direction: Optional direction (e.g. maximize or
                minimize) that indicates which values are considered better.

        Returns:
            The created categorical annotation config.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/409/429).
        """
        from arize._generated import api_client as gen

        space_id = _find_space_id(self._spaces_api, space)
        body = gen.CreateAnnotationConfigRequest(
            actual_instance=gen.CreateCategoricalAnnotationConfigRequest(
                name=name,
                space_id=space_id,
                annotation_config_type=AnnotationConfigType.CATEGORICAL.value,
                values=values,
                optimization_direction=optimization_direction,
            )
        )
        result = self._api.create_annotation_config(
            create_annotation_config_request=body
        )
        return unwrap_oneof(result)  # type: ignore[return-value]

    @prerelease_endpoint(
        key="annotation_configs.create", stage=ReleaseStage.BETA
    )
    def create_freeform(
        self,
        *,
        name: str,
        space: str,
    ) -> FreeformAnnotationConfig:
        """Create a freeform annotation config.

        Freeform annotation configs let a scorer leave open-ended text
        feedback with no predefined scale or set of values.

        Args:
            name: Annotation config name (must be unique within the target space).
            space: Space ID or name to create the annotation config in.

        Returns:
            The created freeform annotation config.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/409/429).
        """
        from arize._generated import api_client as gen

        space_id = _find_space_id(self._spaces_api, space)
        body = gen.CreateAnnotationConfigRequest(
            actual_instance=gen.CreateFreeformAnnotationConfigRequest(
                name=name,
                space_id=space_id,
                annotation_config_type=AnnotationConfigType.FREEFORM.value,
            )
        )
        result = self._api.create_annotation_config(
            create_annotation_config_request=body
        )
        return unwrap_oneof(result)  # type: ignore[return-value]

    @prerelease_endpoint(key="annotation_configs.get", stage=ReleaseStage.BETA)
    def get(
        self, *, annotation_config: str, space: str | None = None
    ) -> (
        CategoricalAnnotationConfig
        | ContinuousAnnotationConfig
        | FreeformAnnotationConfig
    ):
        """Get an annotation config by ID or name.

        Args:
            annotation_config: Annotation config ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_config* is a
                name so it can be resolved to an ID.

        Returns:
            The annotation config object.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        annotation_config_id = _find_annotation_config_id(
            api=self._api,
            annotation_config=annotation_config,
            space=space,
        )
        result = self._api.get_annotation_config(
            annotation_config_id=annotation_config_id
        )
        return unwrap_oneof(result)  # type: ignore[return-value]

    @prerelease_endpoint(
        key="annotation_configs.update", stage=ReleaseStage.BETA
    )
    def update_continuous(
        self,
        *,
        annotation_config: str,
        space: str | None = None,
        name: str | None = None,
        minimum_score: float | int | None = None,
        maximum_score: float | int | None = None,
        optimization_direction: OptimizationDirection | None = None,
    ) -> ContinuousAnnotationConfig:
        """Update a continuous annotation config by ID or name.

        Only the fields you pass are changed; omitted fields are left
        unchanged. The stored config must already be of type `continuous`.

        Args:
            annotation_config: Annotation config ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_config* is a
                name so it can be resolved to an ID.
            name: New name for the annotation config. Must be unique within
                the space.
            minimum_score: New minimum score.
            maximum_score: New maximum score.
            optimization_direction: New optimization direction.

        Returns:
            The updated annotation config object as returned by the API.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/404/409/422/429).
        """
        from arize._generated import api_client as gen

        annotation_config_id = _find_annotation_config_id(
            api=self._api,
            annotation_config=annotation_config,
            space=space,
        )
        body = gen.UpdateAnnotationConfigRequest(
            actual_instance=gen.UpdateContinuousAnnotationConfigRequest(
                annotation_config_type=AnnotationConfigType.CONTINUOUS.value,
                name=name,
                minimum_score=minimum_score,
                maximum_score=maximum_score,
                optimization_direction=optimization_direction,
            )
        )
        result = self._api.update_annotation_config(
            annotation_config_id=annotation_config_id,
            update_annotation_config_request=body,
        )
        return unwrap_oneof(result)  # type: ignore[return-value]

    @prerelease_endpoint(
        key="annotation_configs.update", stage=ReleaseStage.BETA
    )
    def update_categorical(
        self,
        *,
        annotation_config: str,
        space: str | None = None,
        name: str | None = None,
        values: builtins.list[CategoricalAnnotationValue] | None = None,
        optimization_direction: OptimizationDirection | None = None,
    ) -> CategoricalAnnotationConfig:
        """Update a categorical annotation config by ID or name.

        Only the fields you pass are changed; omitted fields are left
        unchanged. The stored config must already be of type `categorical`.

        Args:
            annotation_config: Annotation config ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_config* is a
                name so it can be resolved to an ID.
            name: New name for the annotation config. Must be unique within
                the space.
            values: Replacement set of categorical values. Replaces the full
                label set.
            optimization_direction: New optimization direction.

        Returns:
            The updated annotation config object as returned by the API.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/404/409/422/429).
        """
        from arize._generated import api_client as gen

        annotation_config_id = _find_annotation_config_id(
            api=self._api,
            annotation_config=annotation_config,
            space=space,
        )
        body = gen.UpdateAnnotationConfigRequest(
            actual_instance=gen.UpdateCategoricalAnnotationConfigRequest(
                annotation_config_type=AnnotationConfigType.CATEGORICAL.value,
                name=name,
                values=values,
                optimization_direction=optimization_direction,
            )
        )
        result = self._api.update_annotation_config(
            annotation_config_id=annotation_config_id,
            update_annotation_config_request=body,
        )
        return unwrap_oneof(result)  # type: ignore[return-value]

    @prerelease_endpoint(
        key="annotation_configs.update", stage=ReleaseStage.BETA
    )
    def update_freeform(
        self,
        *,
        annotation_config: str,
        space: str | None = None,
        name: str | None = None,
    ) -> FreeformAnnotationConfig:
        """Update a freeform annotation config by ID or name.

        Only the fields you pass are changed; omitted fields are left
        unchanged. The stored config must already be of type `freeform`.

        Args:
            annotation_config: Annotation config ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_config* is a
                name so it can be resolved to an ID.
            name: New name for the annotation config. Must be unique within
                the space.

        Returns:
            The updated annotation config object as returned by the API.

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/404/409/422/429).
        """
        from arize._generated import api_client as gen

        annotation_config_id = _find_annotation_config_id(
            api=self._api,
            annotation_config=annotation_config,
            space=space,
        )
        body = gen.UpdateAnnotationConfigRequest(
            actual_instance=gen.UpdateFreeformAnnotationConfigRequest(
                annotation_config_type=AnnotationConfigType.FREEFORM.value,
                name=name,
            )
        )
        result = self._api.update_annotation_config(
            annotation_config_id=annotation_config_id,
            update_annotation_config_request=body,
        )
        return unwrap_oneof(result)  # type: ignore[return-value]

    @prerelease_endpoint(
        key="annotation_configs.delete", stage=ReleaseStage.BETA
    )
    def delete(
        self, *, annotation_config: str, space: str | None = None
    ) -> None:
        """Delete an annotation config by ID or name.

        This operation is irreversible.

        Args:
            annotation_config: Annotation config ID or name. If a name is
                provided, *space* is required for resolution.
            space: Space ID or name. Required when *annotation_config* is a
                name so it can be resolved to an ID.

        Returns:
            This method returns None on success (common empty 204 response).

        Raises:
            ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        annotation_config_id = _find_annotation_config_id(
            api=self._api,
            annotation_config=annotation_config,
            space=space,
        )
        return self._api.delete_annotation_config(
            annotation_config_id=annotation_config_id
        )
