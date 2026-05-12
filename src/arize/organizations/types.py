"""Public type re-exports and SDK-facing role types for the organizations subdomain."""

from __future__ import annotations

from dataclasses import dataclass

from arize._generated.api_client.models.organization import Organization
from arize._generated.api_client.models.organization_custom_role_assignment import (
    OrganizationCustomRoleAssignment as _CustomOrgGenerated,
)
from arize._generated.api_client.models.organization_membership import (
    OrganizationMembership,
)
from arize._generated.api_client.models.organization_membership_input import (
    OrganizationMembershipInput,
)
from arize._generated.api_client.models.organization_predefined_role_assignment import (
    OrganizationPredefinedRoleAssignment as _PredefinedOrgGenerated,
)
from arize._generated.api_client.models.organization_role import (
    OrganizationRole,
)
from arize._generated.api_client.models.organization_role_assignment import (
    OrganizationRoleAssignment,
)
from arize._generated.api_client.models.organization_role_assignment_type import (
    OrganizationRoleAssignmentType,
)
from arize._generated.api_client.models.organizations_list200_response import (
    OrganizationsList200Response,
)


@dataclass
class CustomOrgRole:
    """A custom RBAC role assignment for an organization.

    The ``type`` discriminator is set to ``"custom"`` automatically.

    Args:
        id: The unique identifier of the custom RBAC role.
    """

    id: str

    def _to_generated(self) -> _CustomOrgGenerated:
        return _CustomOrgGenerated(
            type=OrganizationRoleAssignmentType.CUSTOM,
            id=self.id,
        )


@dataclass
class PredefinedOrgRole:
    """A predefined organization role assignment.

    The ``type`` discriminator is set to ``"predefined"`` automatically.

    Args:
        name: The predefined role name (``"admin"``, ``"member"``,
            ``"read-only"``, or ``"annotator"``).
    """

    name: OrganizationRole

    def _to_generated(self) -> _PredefinedOrgGenerated:
        return _PredefinedOrgGenerated(
            type=OrganizationRoleAssignmentType.PREDEFINED,
            name=self.name,
        )


__all__ = [
    "CustomOrgRole",
    "Organization",
    "OrganizationMembership",
    "OrganizationMembershipInput",
    "OrganizationRole",
    "OrganizationRoleAssignment",
    "OrganizationsList200Response",
    "PredefinedOrgRole",
]
