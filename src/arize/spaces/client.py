"""Client implementation for managing spaces in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import _find_space_id

if TYPE_CHECKING:
    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration
    from arize.spaces.types import (
        CustomSpaceRole,
        PredefinedSpaceRole,
        Space,
        SpaceMembership,
        SpacesList200Response,
    )

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
        name: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> SpacesList200Response:
        """List spaces the user has access to.

        This endpoint supports cursor-based pagination. When provided,
        ``organization_id`` filters results to a particular organization.
        ``name`` filters results to spaces whose name contains the given
        substring (case-insensitive).

        Args:
            organization_id: Optional organization ID to filter results.
            name: Optional case-insensitive substring filter on space name.
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
            name=name,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="spaces.get", stage=ReleaseStage.BETA)
    def get(self, *, space: str) -> Space:
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
    ) -> Space:
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
    ) -> Space:
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

    @prerelease_endpoint(key="spaces.add_user", stage=ReleaseStage.ALPHA)
    def add_user(
        self,
        *,
        space: str,
        user_id: str,
        role: PredefinedSpaceRole | CustomSpaceRole,
    ) -> SpaceMembership:
        """Add a user to a space (or update their role if already a member).

        If the user is already a member of the space, their role is updated to
        the specified value (upsert). The user must already be a member of the
        space's parent organization — auto-enrollment is not performed.

        **Role constraints**

        - Users with an ``annotator`` account role can only be assigned the
          ``annotator`` predefined space role.
        - Users with a non-annotator account role cannot be assigned the
          ``annotator`` predefined space role.

        Requires space admin role for predefined roles, or ``ROLE_BINDING_CREATE``
        permission for custom roles.

        Args:
            space: Space ID or name.
            user_id: Global ID of the user to add.
            role: Role assignment for the user. Use
                ``PredefinedSpaceRole(name="<role>")`` for predefined roles
                (``admin``, ``member``, ``read-only``, ``annotator``), or
                ``CustomRoleAssignment(type="custom", id="<role_id>")`` for a
                custom RBAC role.

        Returns:
            The created or updated space membership record.

        Raises:
            ApiException: If the API request fails
                (for example, space or user not found, user not in the parent
                organization, role constraint violation, or insufficient
                permissions).
        """
        space_id = _find_space_id(self._api, space)

        from arize._generated import api_client as gen
        from arize.spaces.types import (
            PredefinedSpaceRole as _PredefinedSpaceRole,
        )

        body = gen.SpaceMembershipInput(
            user_id=user_id,
            role=gen.SpaceRoleAssignment(role._to_generated())
            if isinstance(role, _PredefinedSpaceRole)
            else gen.SpaceRoleAssignment(role),
        )
        return self._api.spaces_add_user(
            space_id=space_id, space_membership_input=body
        )

    @prerelease_endpoint(key="spaces.remove_user", stage=ReleaseStage.ALPHA)
    def remove_user(self, *, space: str, user_id: str) -> None:
        """Remove a user from a space.

        Removes both the legacy space-membership row and any RBAC role bindings
        for the user on this space.

        Requires space admin role (legacy auth) or ``ROLE_BINDING_DELETE``
        permission (RBAC).

        Args:
            space: Space ID or name.
            user_id: Global ID of the user to remove.

        Returns:
            This method returns None on success (204 No Content response).

        Raises:
            ApiException: If the API request fails
                (for example, space or user not found, or user is not a member
                of the space).
        """
        space_id = _find_space_id(self._api, space)
        return self._api.spaces_remove_user(space_id=space_id, user_id=user_id)
