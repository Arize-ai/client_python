"""Public type re-exports and SDK-facing role types for the spaces subdomain."""

from __future__ import annotations

from dataclasses import dataclass

from arize._generated.api_client.models.custom_role_assignment import (
    CustomRoleAssignment as _CustomSpaceGenerated,
)
from arize._generated.api_client.models.predefined_role_assignment import (
    PredefinedRoleAssignment as _PredefinedSpaceGenerated,
)
from arize._generated.api_client.models.space import Space
from arize._generated.api_client.models.space_membership import SpaceMembership
from arize._generated.api_client.models.space_membership_input import (
    SpaceMembershipInput,
)
from arize._generated.api_client.models.space_role_assignment import (
    SpaceRoleAssignment,
)
from arize._generated.api_client.models.space_role_assignment_type import (
    SpaceRoleAssignmentType,
)
from arize._generated.api_client.models.spaces_list200_response import (
    SpacesList200Response,
)
from arize._generated.api_client.models.user_space_role import UserSpaceRole


@dataclass
class CustomSpaceRole:
    """A custom RBAC role assignment for a space.

    The ``type`` discriminator is set to ``"custom"`` automatically.

    Args:
        id: The unique identifier of the custom RBAC role.
    """

    id: str

    def _to_generated(self) -> _CustomSpaceGenerated:
        return _CustomSpaceGenerated(
            type=SpaceRoleAssignmentType.CUSTOM,
            id=self.id,
        )


@dataclass
class PredefinedSpaceRole:
    """A predefined space role assignment.

    The ``type`` discriminator is set to ``"predefined"`` automatically.

    Args:
        name: The predefined role name (``"admin"``, ``"member"``,
            ``"read-only"``, or ``"annotator"``).
    """

    name: UserSpaceRole

    def _to_generated(self) -> _PredefinedSpaceGenerated:
        return _PredefinedSpaceGenerated(
            type=SpaceRoleAssignmentType.PREDEFINED,
            name=self.name,
        )


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
