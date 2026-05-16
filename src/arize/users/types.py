"""Public type re-exports and SDK-facing role types for the users subdomain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from arize._generated.api_client.models.create_user_request import (
    CreateUserRequest,
)
from arize._generated.api_client.models.custom_user_role_assignment import (
    CustomUserRoleAssignment,
)
from arize._generated.api_client.models.invite_mode import InviteMode
from arize._generated.api_client.models.predefined_user_role_assignment import (
    PredefinedUserRoleAssignment,
)

if TYPE_CHECKING:
    from datetime import datetime

    from arize._generated.api_client.models.pagination_metadata import (
        PaginationMetadata,
    )
from arize._generated.api_client.models.user_created_response import (
    UserCreatedResponse,
)
from arize._generated.api_client.models.user_role import UserRole
from arize._generated.api_client.models.user_role_assignment import (
    UserRoleAssignment,
)
from arize._generated.api_client.models.user_status import UserStatus
from arize._generated.api_client.models.user_update import UserUpdate


class PredefinedUserRole(PredefinedUserRoleAssignment):
    """A predefined account-level role assignment.

    The ``type`` discriminator is set to ``"predefined"`` automatically.

    Args:
        name: The predefined role name (``"admin"``, ``"member"``, or
            ``"annotator"``).
    """

    type: Literal["predefined"] = "predefined"  # type: ignore[assignment]

    def __str__(self) -> str:
        """Return the role name as a string."""
        return self.name.value


class CustomUserRole(CustomUserRoleAssignment):
    """A custom RBAC role assignment.

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


_UserRoleField = Annotated[
    PredefinedUserRole | CustomUserRole,
    Field(discriminator="type"),
]


class User(BaseModel):
    """An account user with domain-typed role."""

    id: str
    name: str
    email: str
    created_at: datetime
    status: UserStatus
    role: _UserRoleField
    is_developer: bool

    @field_validator("role", mode="before")
    @classmethod
    def _coerce_role(cls, v: object) -> object:
        if isinstance(v, UserRoleAssignment):
            actual = v.actual_instance
            if isinstance(actual, PredefinedUserRoleAssignment):
                return PredefinedUserRole(name=actual.name)
            if isinstance(actual, CustomUserRoleAssignment):
                return CustomUserRole(id=actual.id, name=actual.name)
            raise TypeError(f"Unknown role type: {type(actual)!r}")
        return v


class UsersList200Response(BaseModel):
    """Paginated list of users with domain-typed roles."""

    users: list[User]
    pagination: PaginationMetadata


class DeletionStatus(str, Enum):
    """Outcome of a single user deletion attempt."""

    DELETED = "deleted"
    FAILED = "failed"
    NOT_FOUND = "not_found"


@dataclass
class BulkUserDeletionResult:
    """Result of a single user deletion attempt.

    Attributes:
        id: User ID, or the email address if the user could not
            be resolved.
        status: Outcome of the deletion attempt.
        error: Error message when ``status`` is ``"failed"`` or
            ``"not_found"``.
    """

    id: str
    status: DeletionStatus
    error: str | None = None


__all__ = [
    "BulkUserDeletionResult",
    "CreateUserRequest",
    "CustomUserRole",
    "DeletionStatus",
    "InviteMode",
    "PredefinedUserRole",
    "User",
    "UserCreatedResponse",
    "UserRole",
    "UserStatus",
    "UserUpdate",
    "UsersList200Response",
]
