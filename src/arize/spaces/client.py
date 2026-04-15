"""Client implementation for managing spaces in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import _find_space_id

if TYPE_CHECKING:
    from arize._generated.api_client import models
    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration

logger = logging.getLogger(__name__)


class SpacesClient:
    """Client for managing Arize spaces.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The spaces client is a thin wrapper around the generated REST API client,
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
        self._api = gen.SpacesApi(generated_client)

    @prerelease_endpoint(key="spaces.list", stage=ReleaseStage.BETA)
    def list(
        self,
        *,
        organization_id: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> models.SpacesList200Response:
        """List spaces the user has access to.

        This endpoint supports cursor-based pagination. When provided,
        ``organization_id`` filters results to a particular organization.

        Args:
            organization_id: Optional organization ID to filter results.
            limit: Maximum number of spaces to return. The server may enforce
                an upper bound.
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated space list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails.
        """
        return self._api.spaces_list(
            org_id=organization_id,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="spaces.get", stage=ReleaseStage.BETA)
    def get(self, *, space: str) -> models.Space:
        """Get a space by ID or name.

        Args:
            space: Space ID or name to retrieve.

        Returns:
            The space object.

        Raises:
            ApiException: If the API request fails
                (for example, space not found).
        """
        space_id = _find_space_id(self._api, space)
        return self._api.spaces_get(space_id=space_id)

    @prerelease_endpoint(key="spaces.create", stage=ReleaseStage.BETA)
    def create(
        self,
        *,
        name: str,
        organization_id: str,
        description: str | None = None,
    ) -> models.Space:
        """Create a new space.

        Space names must be unique within the target organization.

        Args:
            name: Space name (must be unique within ``organization_id``).
            organization_id: Organization ID to create the space in.
            description: Optional description of the space's purpose.

        Returns:
            The created space object.

        Raises:
            ApiException: If the API request fails.
        """
        from arize._generated import api_client as gen

        body = gen.SpacesCreateRequest(
            name=name,
            organization_id=organization_id,
            description=description,
        )
        return self._api.spaces_create(spaces_create_request=body)

    @prerelease_endpoint(key="spaces.delete", stage=ReleaseStage.ALPHA)
    def delete(self, *, space: str) -> None:
        """Delete a space by ID or name.

        This operation is irreversible. It deletes the space and all resources
        that belong to it (models, monitors, dashboards, datasets, custom
        metrics, etc).

        Args:
            space: Space ID or name to delete.

        Returns:
            This method returns None on success (204 No Content response).

        Raises:
            ApiException: If the API request fails
                (for example, space not found or insufficient permissions).
        """
        space_id = _find_space_id(self._api, space)
        return self._api.spaces_delete(space_id=space_id)

    @prerelease_endpoint(key="spaces.update", stage=ReleaseStage.BETA)
    def update(
        self,
        *,
        space: str,
        name: str | None = None,
        description: str | None = None,
    ) -> models.Space:
        """Update a space by ID or name.

        Args:
            space: Space ID or name to update.
            name: Updated name for the space.
            description: Updated description for the space.

        Returns:
            The updated space object.

        Raises:
            ValueError: If neither ``name`` nor ``description`` is provided.
            ApiException: If the API request fails
                (for example, space not found or insufficient permissions).
        """
        if name is None and description is None:
            raise ValueError(
                "At least one of 'name' or 'description' must be provided"
            )

        space_id = _find_space_id(self._api, space)

        from arize._generated import api_client as gen

        body = gen.SpacesUpdateRequest(
            name=name,
            description=description,
        )
        return self._api.spaces_update(
            space_id=space_id, spaces_update_request=body
        )
