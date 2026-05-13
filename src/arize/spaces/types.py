"""Public type re-exports and SDK-facing role types for the spaces subdomain."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from arize._generated.api_client.models.custom_role_assignment import (
    CustomRoleAssignment,
)
from arize._generated.api_client.models.predefined_role_assignment import (
    PredefinedRoleAssignment,
)
from arize._generated.api_client.models.space import Space
from arize._generated.api_client.models.space_membership_input import (
    SpaceMembershipInput,
)
from arize._generated.api_client.models.space_role_assignment import (
    SpaceRoleAssignment,
)
from arize._generated.api_client.models.spaces_list200_response import (
    SpacesList200Response,
)
from arize._generated.api_client.models.user_space_role import UserSpaceRole


class PredefinedSpaceRole(PredefinedRoleAssignment):
    """A predefined space role assignment.

    The ``type`` discriminator is set to ``"predefined"`` automatically.

    Args:
        name: The predefined role name (``"admin"``, ``"member"``,
            ``"read-only"``, or ``"annotator"``).
    """

    type: Literal["predefined"] = "predefined"  # type: ignore[assignment]

    def __str__(self) -> str:
        """Return the role name as a string."""
        return self.name.value


class CustomSpaceRole(CustomRoleAssignment):
    """A custom RBAC role assignment for a space.

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


_SpaceRoleField = Annotated[
    PredefinedSpaceRole | CustomSpaceRole,
    Field(discriminator="type"),
]


class SpaceMembership(BaseModel):
    """A space membership record with domain-typed role."""

    id: str
    user_id: str
    space_id: str
    role: _SpaceRoleField

    @field_validator("role", mode="before")
    @classmethod
    def _coerce_role(cls, v: object) -> object:
        if isinstance(v, SpaceRoleAssignment):
            actual = v.actual_instance
            if isinstance(actual, PredefinedRoleAssignment):
                return PredefinedSpaceRole(name=actual.name)
            if isinstance(actual, CustomRoleAssignment):
                return CustomSpaceRole(id=actual.id, name=actual.name)
            raise TypeError(f"Unknown space role type: {type(actual)!r}")
        return v


__all__ = [
    "CustomSpaceRole",
    "PredefinedSpaceRole",
    "Space",
    "SpaceMembership",
    "SpaceMembershipInput",
    "SpaceRoleAssignment",
    "SpacesList200Response",
    "UserSpaceRole",
]
