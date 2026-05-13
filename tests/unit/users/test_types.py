"""Tests for arize.users.types public re-exports."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import arize.users.types as types_module
from arize._generated.api_client.models.pagination_metadata import (
    PaginationMetadata,
)
from arize.users.types import (
    CreateUserRequest,
    CustomUserRole,
    PredefinedUserRole,
    User,
    UserCreatedResponse,
    UserRole,
    UsersList200Response,
    UserStatus,
    UserUpdate,
)

# `datetime` and `PaginationMetadata` are imported only under TYPE_CHECKING in
# users/types.py, so Pydantic v2 cannot resolve them at runtime without help.
User.model_rebuild(_types_namespace={"datetime": datetime})
UsersList200Response.model_rebuild(
    _types_namespace={
        "datetime": datetime,
        "PaginationMetadata": PaginationMetadata,
    }
)


@pytest.mark.unit
class TestUsersTypes:
    """Tests for the users types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "PredefinedUserRole" in types_module.__all__
        assert "CustomUserRole" in types_module.__all__
        assert "CreateUserRequest" in types_module.__all__
        assert "User" in types_module.__all__
        assert "UserCreatedResponse" in types_module.__all__
        assert "UserRole" in types_module.__all__
        assert "UserStatus" in types_module.__all__
        assert "UserUpdate" in types_module.__all__
        assert "UsersList200Response" in types_module.__all__

    def test_user_is_class(self) -> None:
        assert isinstance(User, type)

    def test_create_user_request_is_class(self) -> None:
        assert isinstance(CreateUserRequest, type)

    def test_user_created_response_is_class(self) -> None:
        assert isinstance(UserCreatedResponse, type)

    def test_user_role_is_class(self) -> None:
        assert isinstance(UserRole, type)

    def test_user_status_is_class(self) -> None:
        assert isinstance(UserStatus, type)

    def test_user_update_is_class(self) -> None:
        assert isinstance(UserUpdate, type)

    def test_users_list_response_is_class(self) -> None:
        assert isinstance(UsersList200Response, type)


@pytest.mark.unit
class TestPredefinedUserRole:
    """Tests for the PredefinedUserRole convenience wrapper."""

    def test_is_pydantic_model(self) -> None:
        """PredefinedUserRole should be a Pydantic BaseModel subclass."""
        from pydantic import BaseModel

        assert issubclass(PredefinedUserRole, BaseModel)

    def test_type_field_defaults_to_predefined(self) -> None:
        """Type field should default to 'predefined' without explicit arg."""
        role = PredefinedUserRole(name=UserRole.MEMBER)
        assert role.type == "predefined"

    def test_accepts_all_user_roles(self) -> None:
        """PredefinedUserRole should accept every UserRole enum value."""
        for role_value in UserRole:
            role = PredefinedUserRole(name=role_value)
            assert role.name == role_value


@pytest.mark.unit
class TestCustomUserRole:
    """Tests for the CustomUserRole convenience wrapper."""

    def test_is_pydantic_model(self) -> None:
        """CustomUserRole should be a Pydantic BaseModel subclass."""
        from pydantic import BaseModel

        assert issubclass(CustomUserRole, BaseModel)

    def test_type_field_defaults_to_custom(self) -> None:
        """Type field should default to 'custom' without explicit arg."""
        role = CustomUserRole(id="role-abc-123")
        assert role.type == "custom"

    def test_name_defaults_to_none(self) -> None:
        """Name should be optional, defaulting to None."""
        role = CustomUserRole(id="role-abc-123")
        assert role.name is None

    def test_name_preserved_when_set(self) -> None:
        """Name should be stored when provided."""
        role = CustomUserRole(id="role-abc-123", name="My Custom Role")
        assert role.name == "My Custom Role"


@pytest.mark.unit
class TestUser:
    """Tests for User role coercion via field_validator."""

    def _make_role_assignment(self, actual_instance: object) -> MagicMock:
        from arize._generated.api_client.models.user_role_assignment import (
            UserRoleAssignment,
        )

        assignment = MagicMock(spec=UserRoleAssignment)
        assignment.actual_instance = actual_instance
        return assignment

    def test_coerce_role_predefined(self) -> None:
        from arize._generated.api_client.models.predefined_user_role_assignment import (
            PredefinedUserRoleAssignment,
        )

        predefined = MagicMock(spec=PredefinedUserRoleAssignment)
        predefined.name = UserRole.ADMIN
        assignment = self._make_role_assignment(predefined)

        user = User(
            id="user-42",
            name="Alice",
            email="alice@example.com",
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            status=UserStatus.ACTIVE,
            role=assignment,
            is_developer=True,
        )

        assert isinstance(user.role, PredefinedUserRole)
        assert user.role.name == UserRole.ADMIN

    def test_coerce_role_custom(self) -> None:
        from arize._generated.api_client.models.custom_user_role_assignment import (
            CustomUserRoleAssignment,
        )

        custom = MagicMock(spec=CustomUserRoleAssignment)
        custom.id = "custom-role-5"
        custom.name = "Power User"
        assignment = self._make_role_assignment(custom)

        user = User(
            id="user-42",
            name="Alice",
            email="alice@example.com",
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            status=UserStatus.ACTIVE,
            role=assignment,
            is_developer=True,
        )

        assert isinstance(user.role, CustomUserRole)
        assert user.role.id == "custom-role-5"
        assert user.role.name == "Power User"

    def test_coerce_role_unknown_raises(self) -> None:
        assignment = self._make_role_assignment(object())

        with pytest.raises(Exception, match="Unknown role type"):
            User(
                id="user-42",
                name="Alice",
                email="alice@example.com",
                created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
                status=UserStatus.ACTIVE,
                role=assignment,
                is_developer=True,
            )

    def test_user_fields_preserved(self) -> None:
        from arize._generated.api_client.models.predefined_user_role_assignment import (
            PredefinedUserRoleAssignment,
        )

        predefined = MagicMock(spec=PredefinedUserRoleAssignment)
        predefined.name = UserRole.MEMBER
        assignment = self._make_role_assignment(predefined)

        user = User(
            id="user-42",
            name="Alice",
            email="alice@example.com",
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            status=UserStatus.ACTIVE,
            role=assignment,
            is_developer=True,
        )

        assert user.id == "user-42"
        assert user.name == "Alice"
        assert user.email == "alice@example.com"
        assert user.status == UserStatus.ACTIVE
        assert user.is_developer is True
        assert isinstance(user.role, PredefinedUserRole)
        assert user.role.name == UserRole.MEMBER


@pytest.mark.unit
class TestUsersList200Response:
    """Tests for UsersList200Response."""

    def _make_user(self, role_name: UserRole = UserRole.MEMBER) -> User:
        return User(
            id="u-1",
            name="Bob",
            email="bob@example.com",
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            status=UserStatus.ACTIVE,
            role=PredefinedUserRole(name=role_name),
            is_developer=False,
        )

    def _make_pagination(self) -> PaginationMetadata:
        return PaginationMetadata(has_more=False, next_cursor=None)

    def test_from_generated_maps_users(self) -> None:
        pagination = self._make_pagination()
        users = [self._make_user(), self._make_user(UserRole.ADMIN)]

        result = UsersList200Response(users=users, pagination=pagination)

        assert len(result.users) == 2
        assert result.pagination == pagination
        assert all(isinstance(u, User) for u in result.users)

    def test_from_generated_handles_none_users(self) -> None:
        pagination = self._make_pagination()

        result = UsersList200Response(users=[], pagination=pagination)

        assert result.users == []
        assert result.pagination == pagination
