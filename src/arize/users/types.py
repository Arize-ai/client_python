"""Public type re-exports and SDK-facing role types for the users subdomain."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from arize._generated.api_client.models.create_user_request import (
    CreateUserRequest,
)
from arize._generated.api_client.models.custom_user_role_assignment import (
    CustomUserRoleAssignment,
)
from arize._generated.api_client.models.invite_mode import InviteMode
from arize._generated.api_client.models.pagination_metadata import (
    PaginationMetadata,
)
from arize._generated.api_client.models.predefined_user_role_assignment import (
    PredefinedUserRoleAssignment,
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


class UserListResponse(BaseModel):
    """Paginated list of users with domain-typed roles."""

    users: list[User]
    pagination: PaginationMetadata


class DeletionStatus(str, Enum):
    """Outcome of a single user deletion attempt."""

    DELETED = "deleted"
    FAILED = "failed"
    NOT_FOUND = "not_found"


class BulkUserDeletionResult(BaseModel):
    """Result of a single user deletion attempt.

    Attributes:
        user_id: ID of the user targeted for deletion. Empty when an email
            could not be resolved to a user (status ``"not_found"``).
        email: Email address the deletion was requested for, set only when
            the user was specified by email. ``None`` when specified by ID.
        status: Outcome of the deletion attempt.
        error: Error message when ``status`` is ``"failed"`` or
            ``"not_found"``.
    """

    user_id: str
    status: DeletionStatus
    email: str | None = None
    error: str | None = None


class BulkDeleteResponse(BaseModel):
    """Response from a bulk user delete operation.

    Attributes:
        results: Per-user deletion outcomes.
    """

    results: list[BulkUserDeletionResult]


__all__ = [
    "BulkDeleteResponse",
    "BulkUserDeletionResult",
    "CreateUserRequest",
    "CustomUserRole",
    "DeletionStatus",
    "InviteMode",
    "PredefinedUserRole",
    "User",
    "UserCreatedResponse",
    "UserListResponse",
    "UserRole",
    "UserStatus",
    "UserUpdate",
]
