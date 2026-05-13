"""Public type re-exports and SDK-facing role types for the organizations subdomain."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from arize._generated.api_client.models.organization import Organization
from arize._generated.api_client.models.organization_custom_role_assignment import (
    OrganizationCustomRoleAssignment,
)
from arize._generated.api_client.models.organization_membership_input import (
    OrganizationMembershipInput,
)
from arize._generated.api_client.models.organization_predefined_role_assignment import (
    OrganizationPredefinedRoleAssignment,
)
from arize._generated.api_client.models.organization_role import (
    OrganizationRole,
)
from arize._generated.api_client.models.organization_role_assignment import (
    OrganizationRoleAssignment,
)
from arize._generated.api_client.models.organizations_list200_response import (
    OrganizationsList200Response,
)


class PredefinedOrgRole(OrganizationPredefinedRoleAssignment):
    """A predefined organization role assignment.

    The ``type`` discriminator is set to ``"predefined"`` automatically.

    Args:
        name: The predefined role name (``"admin"``, ``"member"``,
            ``"read-only"``, or ``"annotator"``).
    """

    type: Literal["predefined"] = "predefined"  # type: ignore[assignment]

    def __str__(self) -> str:
        """Return the role name as a string."""
        return self.name.value


class CustomOrgRole(OrganizationCustomRoleAssignment):
    """A custom RBAC role assignment for an organization.

    The ``type`` discriminator is set to ``"custom"`` automatically.

    Args:
        id: The unique identifier of the custom RBAC role.
        name: Human-readable name of the custom role (returned in responses
            only; ignored on input).
    """

    type: Literal["custom"] = "custom"  # type: ignore[assignment]

    def __str__(self) -> str:
        """Return the role name if available, otherwise the role id."""
        return self.name if self.name is not None else self.id


_OrgRoleField = Annotated[
    PredefinedOrgRole | CustomOrgRole,
    Field(discriminator="type"),
]


class OrganizationMembership(BaseModel):
    """An organization membership record with domain-typed role."""

    id: str
    user_id: str
    organization_id: str
    role: _OrgRoleField

    @field_validator("role", mode="before")
    @classmethod
    def _coerce_role(cls, v: object) -> object:
        if isinstance(v, OrganizationRoleAssignment):
            actual = v.actual_instance
            if isinstance(actual, OrganizationPredefinedRoleAssignment):
                return PredefinedOrgRole(name=actual.name)
            if isinstance(actual, OrganizationCustomRoleAssignment):
                return CustomOrgRole(id=actual.id, name=actual.name)
            raise TypeError(f"Unknown org role type: {type(actual)!r}")
        return v


__all__ = [
    "CustomOrgRole",
    "Organization",
    "OrganizationMembership",
    "OrganizationMembershipInput",
    "OrganizationRole",
    "OrganizationsList200Response",
    "PredefinedOrgRole",
]
