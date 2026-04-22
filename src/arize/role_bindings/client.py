"""Client implementation for managing role bindings in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.pre_releases import ReleaseStage, prerelease_endpoint

if TYPE_CHECKING:
    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration
    from arize.role_bindings.types import RoleBinding, RoleBindingResourceType

logger = logging.getLogger(__name__)


class RoleBindingsClient:
    """Client for managing Arize role bindings.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    A role binding assigns a role to a user on a specific resource. Only one
    binding per user per resource is allowed. The ``resource_type`` must be
    either ``SPACE`` or ``PROJECT``, and the ``resource_id`` must be a global ID
    encoding a resource of the matching type.

    The role bindings client is a thin wrapper around the generated REST API
    client, using the shared generated API client owned by
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
        self._api = gen.RoleBindingsApi(generated_client)

    @prerelease_endpoint(key="role_bindings.create", stage=ReleaseStage.ALPHA)
    def create(
        self,
        *,
        user_id: str,
        role_id: str,
        resource_type: RoleBindingResourceType,
        resource_id: str,
    ) -> RoleBinding:
        """Create a new role binding.

        Assigns a role to a user on the specified resource. Only one binding per
        user per resource is allowed.

        # Example:
        # >>> client = arize.role_bindings.Client(...)
        # >>> binding = client.create(
        # ...     user_id="user-123",
        # ...     role_id="role-456",
        # ...     resource_type=RoleBindingResourceType.SPACE,
        # ...     resource_id="space-789"
        # ... )
        # >>> print(binding.role_id)
        # "role-456"

        Args:
            user_id: Global ID of the user to bind the role to.
            role_id: Global ID of the role to assign.
            resource_type: Type of resource to bind the role on
                (``RoleBindingResourceType.SPACE`` or
                ``RoleBindingResourceType.PROJECT``).
            resource_id: Global ID of the resource. Must encode a resource of
                the type specified by ``resource_type``.

        Returns:
            The created role binding object.

        Raises:
            ConflictException: If a binding already exists for the user on the
                specified resource.
            ApiException: If the API request fails (for example, invalid
                resource type/ID combination or insufficient permissions).
        """
        from arize._generated import api_client as gen

        body = gen.RoleBindingCreate(
            user_id=user_id,
            role_id=role_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        return self._api.role_bindings_create(body)

    @prerelease_endpoint(key="role_bindings.get", stage=ReleaseStage.ALPHA)
    def get(self, *, binding_id: str) -> RoleBinding:
        """Get a role binding by ID.

        Args:
            binding_id: Role binding ID to retrieve.

        Returns:
            The role binding object.

        Raises:
            ApiException: If the API request fails
                (for example, binding not found).
        """
        return self._api.role_bindings_get(binding_id)

    @prerelease_endpoint(key="role_bindings.update", stage=ReleaseStage.ALPHA)
    def update(self, *, binding_id: str, role_id: str) -> RoleBinding:
        """Update an existing role binding by replacing its assigned role.

        Only the ``role_id`` can be changed on an existing binding. The user,
        resource type, and resource ID remain the same.

        # Example:
        # >>> client = arize.role_bindings.Client(...)
        # >>> updated_binding = client.update(
        # ...     binding_id="role_binding-123",
        # ...     role_id="role-456"
        # ... )
        # >>> print(updated_binding.role_id)
        # "role-456"

        Args:
            binding_id: Role binding ID to update.
            role_id: New role ID to assign. Replaces the existing role.

        Returns:
            The updated role binding object.

        Raises:
            ApiException: If the API request fails
                (for example, binding not found or insufficient permissions).
        """
        from arize._generated import api_client as gen

        body = gen.RoleBindingUpdate(role_id=role_id)
        return self._api.role_bindings_update(binding_id, body)

    @prerelease_endpoint(key="role_bindings.delete", stage=ReleaseStage.ALPHA)
    def delete(self, *, binding_id: str) -> None:
        """Delete a role binding by ID.

        Args:
            binding_id: Role binding ID to delete.

        Raises:
            ApiException: If the API request fails
                (for example, binding not found or insufficient permissions).
        """
        return self._api.role_bindings_delete(binding_id)
