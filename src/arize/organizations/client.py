"""Client implementation for managing organizations in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.organizations.types import OrganizationMembership
from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import _find_organization_id

if TYPE_CHECKING:
    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration
    from arize.organizations.types import (
        Organization,
        OrganizationsList200Response,
        PredefinedOrgRole,
    )

logger = logging.getLogger(__name__)


class OrganizationsClient:
    """Client for managing Arize organizations.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The organizations client is a thin wrapper around the generated REST API
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
        self._api = gen.OrganizationsApi(generated_client)

    @prerelease_endpoint(key="organizations.list", stage=ReleaseStage.ALPHA)
    def list(
        self,
        *,
        name: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> OrganizationsList200Response:
        """List organizations the user has access to.

        This endpoint supports cursor-based pagination. When provided,
        ``name`` filters results to organizations whose name contains the
        given substring (case-insensitive).

        Args:
            name: Optional case-insensitive substring filter on organization name.
            limit: Maximum number of organizations to return (1-100). The server
                enforces an upper bound of 100.
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated organization list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails.
        """
        return self._api.organizations_list(
            name=name,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="organizations.get", stage=ReleaseStage.ALPHA)
    def get(self, *, organization: str) -> Organization:
        """Get an organization by ID or name.

        Args:
            organization: Organization ID or name to retrieve.

        Returns:
            The organization object.

        Raises:
            ApiException: If the API request fails
                (for example, organization not found).
        """
        org_id = _find_organization_id(self._api, organization)
        return self._api.organizations_get(org_id=org_id)

    @prerelease_endpoint(key="organizations.create", stage=ReleaseStage.ALPHA)
    def create(
        self,
        *,
        name: str,
        description: str | None = None,
    ) -> Organization:
        """Create a new organization.

        Organization names must be unique within the account.

        Args:
            name: Organization name (must be unique within the account).
            description: Optional description of the organization's purpose.

        Returns:
            The created organization object.

        Raises:
            ApiException: If the API request fails.
        """
        from arize._generated import api_client as gen

        body = gen.OrganizationCreate(
            name=name,
            description=description,
        )
        return self._api.organizations_create(organization_create=body)

    @prerelease_endpoint(key="organizations.delete", stage=ReleaseStage.ALPHA)
    def delete(self, *, organization: str) -> None:
        """Delete an organization by ID or name.

        This operation is irreversible. It permanently deletes the
        organization and all resources that belong to it, including all
        spaces and their contents (projects, experiments, evaluators,
        models, monitors, dashboards, datasets, annotation configs,
        annotation queues, custom metrics, etc.) as well as
        organization-level resources (integrations, cost configurations,
        SAML identity providers, and API keys).

        Args:
            organization: Organization ID or name to delete.

        Returns:
            This method returns None on success (204 No Content response).

        Raises:
            ApiException: If the API request fails
                (for example, organization not found or insufficient
                permissions).
        """
        org_id = _find_organization_id(self._api, organization)
        return self._api.organizations_delete(org_id=org_id)

    @prerelease_endpoint(key="organizations.update", stage=ReleaseStage.ALPHA)
    def update(
        self,
        *,
        organization: str,
        name: str | None = None,
        description: str | None = None,
    ) -> Organization:
        """Update an organization's metadata by ID or name.

        Args:
            organization: Organization ID or name to update.
            name: Updated name for the organization.
            description: Updated description for the organization. Pass an
                empty string to clear the existing description.

        Returns:
            The updated organization object.

        Raises:
            ValueError: If neither ``name`` nor ``description`` is provided.
            ApiException: If the API request fails
                (for example, organization not found or insufficient permissions).
        """
        if name is None and description is None:
            raise ValueError(
                "At least one of 'name' or 'description' must be provided"
            )

        org_id = _find_organization_id(self._api, organization)

        from arize._generated import api_client as gen

        body = gen.OrganizationUpdate(
            name=name,
            description=description,
        )
        return self._api.organizations_update(
            org_id=org_id, organization_update=body
        )

    @prerelease_endpoint(key="organizations.add_user", stage=ReleaseStage.ALPHA)
    def add_user(
        self,
        *,
        organization: str,
        user_id: str,
        role: PredefinedOrgRole,
    ) -> OrganizationMembership:
        """Add a user to an organization (or update their role if already a member).

        If the user is already a member of the organization, their role is updated
        to the specified value (upsert).

        **Role constraints**

        - Users with an ``annotator`` account role can only be assigned the
          ``annotator`` organization role.
        - Users with a non-annotator account role cannot be assigned the
          ``annotator`` organization role.

        Requires organization admin.

        Args:
            organization: Organization ID or name.
            user_id: Global ID of the user to add.
            role: Role assignment for the user. Use
                ``PredefinedOrgRole(name="<role>")`` for predefined roles
                (``admin``, ``member``, ``read-only``, ``annotator``).
                Custom role assignments are not yet supported for organizations.

        Returns:
            The created or updated organization membership record.

        Raises:
            ApiException: If the API request fails
                (for example, organization or user not found, role constraint
                violation, or insufficient permissions).
        """
        org_id = _find_organization_id(self._api, organization)

        from arize._generated import api_client as gen

        body = gen.OrganizationMembershipInput(
            user_id=user_id,
            role=gen.OrganizationRoleAssignment(
                gen.OrganizationPredefinedRoleAssignment(
                    type=gen.OrganizationRoleAssignmentType.PREDEFINED,
                    name=role.name,
                )
                if role.type == "predefined"
                else gen.OrganizationCustomRoleAssignment(
                    type=gen.OrganizationRoleAssignmentType.CUSTOM,
                    id=role.id,
                )
            ),
        )
        return OrganizationMembership.model_validate(
            self._api.organizations_add_user(
                org_id=org_id, organization_membership_input=body
            ),
            from_attributes=True,
        )

    @prerelease_endpoint(
        key="organizations.remove_user", stage=ReleaseStage.ALPHA
    )
    def remove_user(self, *, organization: str, user_id: str) -> None:
        """Remove a user from an organization.

        Removes the user from the organization and all its child spaces
        (membership cascade).

        Requires organization admin.

        Args:
            organization: Organization ID or name.
            user_id: Global ID of the user to remove.

        Returns:
            This method returns None on success (204 No Content response).

        Raises:
            ApiException: If the API request fails
                (for example, organization or user not found, or insufficient
                permissions).
        """
        org_id = _find_organization_id(self._api, organization)
        return self._api.organizations_remove_user(
            org_id=org_id, user_id=user_id
        )
