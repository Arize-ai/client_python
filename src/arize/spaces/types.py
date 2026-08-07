"""Public type re-exports and SDK-facing role types for the spaces subdomain.

Domain role types (``PredefinedSpaceRole``, ``CustomSpaceRole``) subclass the
strict ``*Request`` schemas and are used as inputs to write methods. They are
also produced by ``SpaceMembership._coerce_role`` when deserializing responses,
so they serve both as request inputs and as the in-memory representation of a
membership's role field.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from arize._generated.api_client.models.add_space_user_request import (
    AddSpaceUserRequest,
)
from arize._generated.api_client.models.custom_role_assignment import (
    CustomRoleAssignment,
)
from arize._generated.api_client.models.custom_role_assignment_request import (
    CustomRoleAssignmentRequest,
)
from arize._generated.api_client.models.list_spaces_response import (
    ListSpacesResponse,
)
from arize._generated.api_client.models.predefined_role_assignment import (
    PredefinedRoleAssignment,
)
from arize._generated.api_client.models.predefined_role_assignment_request import (
    PredefinedRoleAssignmentRequest,
)
from arize._generated.api_client.models.space import Space
from arize._generated.api_client.models.space_role_assignment import (
    SpaceRoleAssignment,
)
from arize._generated.api_client.models.user_space_role import UserSpaceRole


class PredefinedSpaceRole(PredefinedRoleAssignmentRequest):
    """A predefined space role assignment.

    The ``type`` discriminator is set to ``"PREDEFINED"`` automatically.

    Args:
        name: The predefined role name (``"ADMIN"``, ``"MEMBER"``,
            ``"READ_ONLY"``, or ``"ANNOTATOR"``).
    """

    type: Literal["PREDEFINED"] = "PREDEFINED"  # type: ignore[assignment]

    def __str__(self) -> str:
        """Return the role name as a string."""
        return self.name.value


class CustomSpaceRole(CustomRoleAssignmentRequest):
    """A custom RBAC role assignment for a space.

    The ``type`` discriminator is set to ``"CUSTOM"`` automatically.

    Args:
        id: The unique identifier of the custom RBAC role.
        name: Human-readable name of the custom role (returned in responses
            only; ignored on input).
    """

    # name is a response-only field not present in CustomRoleAssignmentRequest;
    # carried here so __str__ can display it when available.
    name: str | None = None
    type: Literal["CUSTOM"] = "CUSTOM"  # type: ignore[assignment]

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
    "AddSpaceUserRequest",
    "CustomSpaceRole",
    "ListSpacesResponse",
    "PredefinedSpaceRole",
    "Space",
    "SpaceMembership",
    "SpaceRoleAssignment",
    "UserSpaceRole",
]
