"""Client implementation for managing annotation configs in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.annotation_configs.types import AnnotationConfigType
from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import (
    _find_annotation_config_id,
    _find_space_id,
    _resolve_resource,
)

if TYPE_CHECKING:
    import builtins

    from arize._generated.api_client import models
    from arize._generated.api_client.api_client import ApiClient
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
        limit: int = 100,
        cursor: str | None = None,
    ) -> models.AnnotationConfigsList200Response:
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
        return self._api.annotation_configs_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=name,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(
        key="annotation_configs.create", stage=ReleaseStage.BETA
    )
    def create(
        self,
        *,
        name: str,
        space: str,
        config_type: AnnotationConfigType,
        minimum_score: float | int | None = None,
        maximum_score: float | int | None = None,
        values: builtins.list[models.CategoricalAnnotationValue] | None = None,
        optimization_direction: models.OptimizationDirection | None = None,
    ) -> models.AnnotationConfig:
        """Create an annotation config.

        Supported config types:
            - `continuous` requires `minimum_score` and `maximum_score`
            - `categorical` requires `values`
            - `freeform` requires no additional fields

        Args:
            name: Annotation config name (must be unique within the target space).
            space: Space ID or name to create the annotation config in.
            config_type: Type of annotation config to create.
            minimum_score: Minimum score for continuous configs.
            maximum_score: Maximum score for continuous configs.
            values: Categorical values for categorical configs.
            optimization_direction: Optional optimization direction for
                continuous and categorical configs.

        Returns:
            The created annotation config object as returned by the API.

        Raises:
            ValueError: If required fields for the selected config type are missing.
            ApiException: If the REST API
                returns an error response (e.g. 400/401/403/409/429).
        """
        from arize._generated import api_client as gen

        space_id = _find_space_id(self._spaces_api, space)

        if config_type == AnnotationConfigType.CONTINUOUS:
            if minimum_score is None or maximum_score is None:
                raise ValueError(
                    "minimum_score and maximum_score are required for continuous configs"
                )
            body = gen.CreateAnnotationConfigRequestBody(
                actual_instance=gen.ContinuousAnnotationConfigCreate(
                    name=name,
                    space_id=space_id,
                    annotation_config_type=AnnotationConfigType.CONTINUOUS.value,
                    minimum_score=minimum_score,
                    maximum_score=maximum_score,
                    optimization_direction=optimization_direction,
                )
            )
        elif config_type == AnnotationConfigType.CATEGORICAL:
            if values is None:
                raise ValueError("values are required for categorical configs")
            body = gen.CreateAnnotationConfigRequestBody(
                actual_instance=gen.CategoricalAnnotationConfigCreate(
                    name=name,
                    space_id=space_id,
                    annotation_config_type=AnnotationConfigType.CATEGORICAL.value,
                    values=values,
                    optimization_direction=optimization_direction,
                )
            )
        else:
            body = gen.CreateAnnotationConfigRequestBody(
                actual_instance=gen.FreeformAnnotationConfigCreate(
                    name=name,
                    space_id=space_id,
                    annotation_config_type=AnnotationConfigType.FREEFORM.value,
                )
            )
        return self._api.annotation_configs_create(
            create_annotation_config_request_body=body
        )

    @prerelease_endpoint(key="annotation_configs.get", stage=ReleaseStage.BETA)
    def get(
        self, *, annotation_config: str, space: str | None = None
    ) -> models.AnnotationConfig:
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
        return self._api.annotation_configs_get(
            annotation_config_id=annotation_config_id
        )

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
        return self._api.annotation_configs_delete(
            annotation_config_id=annotation_config_id
        )
