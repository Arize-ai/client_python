"""Public type re-exports and SDK-facing role types for the users subdomain."""

from __future__ import annotations

from dataclasses import dataclass

from arize._generated.api_client.models.create_user_request import (
    CreateUserRequest,
)
from arize._generated.api_client.models.custom_user_role_assignment import (
    CustomUserRoleAssignment as _CustomGenerated,
)
from arize._generated.api_client.models.invite_mode import InviteMode
from arize._generated.api_client.models.predefined_user_role_assignment import (
    PredefinedUserRoleAssignment as _PredefinedGenerated,
)
from arize._generated.api_client.models.user import User
from arize._generated.api_client.models.user_created_response import (
    UserCreatedResponse,
)
from arize._generated.api_client.models.user_role import UserRole
from arize._generated.api_client.models.user_role_assignment import (
    UserRoleAssignment,
)
from arize._generated.api_client.models.user_role_assignment_type import (
    UserRoleAssignmentType,
)
from arize._generated.api_client.models.user_status import UserStatus
from arize._generated.api_client.models.user_update import UserUpdate
from arize._generated.api_client.models.users_list200_response import (
    UsersList200Response,
)


@dataclass
class PredefinedUserRole:
    """A predefined account-level role assignment.

    The ``type`` discriminator is set to ``"predefined"`` automatically.

    Args:
        name: The predefined role name (``"admin"``, ``"member"``, or
            ``"annotator"``).
    """

    name: UserRole

    def _to_generated(self) -> _PredefinedGenerated:
        return _PredefinedGenerated(
            type=UserRoleAssignmentType.PREDEFINED,
            name=self.name,
        )


@dataclass
class CustomUserRole:
    """A custom RBAC role assignment.

    The ``type`` discriminator is set to ``"custom"`` automatically.

    Args:
        id: The unique identifier of the custom RBAC role.
    """

    id: str

    def _to_generated(self) -> _CustomGenerated:
        return _CustomGenerated(
            type=UserRoleAssignmentType.CUSTOM,
            id=self.id,
        )


__all__ = [
    "CreateUserRequest",
    "CustomUserRole",
    "InviteMode",
    "PredefinedUserRole",
    "User",
    "UserCreatedResponse",
    "UserRole",
    "UserRoleAssignment",
    "UserStatus",
    "UserUpdate",
    "UsersList200Response",
]
