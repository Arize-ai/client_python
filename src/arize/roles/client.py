"""Client implementation for managing roles in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.pre_releases import ReleaseStage, prerelease_endpoint

if TYPE_CHECKING:
    # builtins is needed for builtins.list in annotations because this class
    # defines list() which shadows the built-in list type.
    import builtins

    from arize._generated.api_client import models
    from arize._generated.api_client.api_client import ApiClient
    from arize._generated.api_client.models.permission import Permission
    from arize.config import SDKConfiguration

logger = logging.getLogger(__name__)


class RolesClient:
    """Client for managing Arize roles.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The roles client is a thin wrapper around the generated REST API client,
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
        self._api = gen.RolesApi(generated_client)

    @prerelease_endpoint(key="roles.list", stage=ReleaseStage.ALPHA)
    def list(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
        is_predefined: bool | None = None,
    ) -> models.RolesList200Response:
        """List roles for the authenticated user's account.

        This endpoint supports cursor-based pagination. When provided,
        ``is_predefined`` filters results to predefined or custom roles only.

        Args:
            limit: Maximum number of roles to return. The server enforces
                an upper bound of 100.
            cursor: Opaque pagination cursor from a previous response.
            is_predefined: When ``True``, return only system-defined predefined
                roles. When ``False``, return only custom (account-defined) roles.
                When ``None`` (default), return all roles.

        Returns:
            A paginated role list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails.
        """
        return self._api.roles_list(
            limit=limit,
            cursor=cursor,
            is_predefined=is_predefined,
        )

    @prerelease_endpoint(key="roles.get", stage=ReleaseStage.ALPHA)
    def get(self, *, role_id: str) -> models.Role:
        """Get a role by ID.

        Args:
            role_id: Role ID to retrieve.

        Returns:
            The role object.

        Raises:
            ApiException: If the API request fails
                (for example, role not found).
        """
        return self._api.roles_get(role_id=role_id)

    @prerelease_endpoint(key="roles.create", stage=ReleaseStage.ALPHA)
    def create(
        self,
        *,
        name: str,
        permissions: builtins.list[Permission],
        description: str | None = None,
    ) -> models.Role:
        """Create a new custom role.

        Role names must be unique within the account. At least one permission
        must be provided. Predefined (system-managed) roles cannot be created
        through this endpoint.

        Args:
            name: Role name (must be unique within the account, max 255 chars).
            permissions: List of permissions to grant (e.g.
                ``[Permission.PROJECT_READ, Permission.DATASET_CREATE]``). At least one is required.
            description: Optional description of the role's purpose (max 1000 chars).

        Returns:
            The created role object.

        Raises:
            ApiException: If the API request fails.
        """
        from arize._generated import api_client as gen

        body = gen.RoleCreate(
            name=name,
            permissions=permissions,
            description=description,
        )
        return self._api.roles_create(role_create=body)

    @prerelease_endpoint(key="roles.update", stage=ReleaseStage.ALPHA)
    def update(
        self,
        *,
        role_id: str,
        name: str | None = None,
        description: str | None = None,
        permissions: builtins.list[Permission] | None = None,
    ) -> models.Role:
        """Update a custom role by ID.

        At least one field must be provided. Predefined roles cannot be updated.
        When ``permissions`` is provided, the existing permissions are fully
        replaced with the new set.

        Args:
            role_id: Role ID to update.
            name: Updated name for the role (max 255 chars).
            description: Updated description of the role (max 1000 chars).
            permissions: Replacement set of permissions. When provided, fully
                replaces existing permissions.

        Returns:
            The updated role object.

        Raises:
            ValueError: If none of ``name``, ``description``, or ``permissions``
                is provided.
            ApiException: If the API request fails
                (for example, role not found, insufficient permissions, or
                attempting to update a predefined role).
        """
        if name is None and description is None and permissions is None:
            raise ValueError(
                "At least one of 'name', 'description', or 'permissions' must be provided"
            )

        from arize._generated import api_client as gen

        body = gen.RoleUpdate(
            name=name,
            description=description,
            permissions=permissions,
        )
        return self._api.roles_update(role_id=role_id, role_update=body)

    @prerelease_endpoint(key="roles.delete", stage=ReleaseStage.ALPHA)
    def delete(self, *, role_id: str) -> None:
        """Delete a custom role by ID.

        Predefined (system-managed) roles cannot be deleted.

        Args:
            role_id: Role ID to delete.

        Raises:
            ApiException: If the API request fails
                (for example, role not found, insufficient permissions, or
                attempting to delete a predefined role).
        """
        return self._api.roles_delete(role_id=role_id)
