"""Client implementation for managing annotation configs in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.annotation_configs.types import AnnotationConfigType
from arize.pre_releases import ReleaseStage, prerelease_endpoint

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

    @prerelease_endpoint(key="annotation_configs.list", stage=ReleaseStage.BETA)
    def list(
        self,
        *,
        space_id: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> models.AnnotationConfigsList200Response:
        """List annotation configs the user has access to.

        Annotation configs are returned in descending creation order (most recently created
        first).

        Args:
            space_id: Optional space ID to scope results to a single space.
            limit: Maximum number of annotation configs to return. The server enforces an
                upper bound.
            cursor: Opaque pagination cursor returned from a previous response.

        Returns:
            A response object with the annotation configs and pagination information.

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 401/403/429).
        """
        return self._api.annotation_configs_list(
            space_id=space_id,
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
        space_id: str,
        type: AnnotationConfigType,
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
            space_id: Space ID to create the annotation config in.
            type: Type of annotation config to create.
            minimum_score: Minimum score for continuous configs.
            maximum_score: Maximum score for continuous configs.
            values: Categorical values for categorical configs.
            optimization_direction: Optional optimization direction for
                continuous and categorical configs.

        Returns:
            The created annotation config object as returned by the API.

        Raises:
            ValueError: If required fields for the selected config type are missing.
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 400/401/403/409/429).
        """
        from arize._generated import api_client as gen

        if type == AnnotationConfigType.CONTINUOUS.value:
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
        elif type == AnnotationConfigType.CATEGORICAL.value:
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
    def get(self, *, id: str) -> models.AnnotationConfig:
        """Get an annotation config by ID.

        Args:
            id: Annotation config ID to retrieve.

        Returns:
            The annotation config object.

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        return self._api.annotation_configs_get(annotation_config_id=id)

    @prerelease_endpoint(
        key="annotation_configs.delete", stage=ReleaseStage.BETA
    )
    def delete(self, *, id: str) -> None:
        """Delete an annotation config by ID.

        This operation is irreversible.

        Args:
            id: Annotation config ID to delete.

        Returns:
            This method returns None on success (common empty 204 response).

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the REST API
                returns an error response (e.g. 401/403/404/429).
        """
        return self._api.annotation_configs_delete(annotation_config_id=id)
