"""Tests for arize.users.types public re-exports."""

from __future__ import annotations

import pytest

import arize.users.types as types_module
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

    def test_to_generated_sets_predefined_type(self) -> None:
        """_to_generated() should return a PredefinedUserRoleAssignment with type=predefined."""
        role = PredefinedUserRole(name=UserRole.MEMBER)
        generated = role._to_generated()
        assert generated.type == "predefined"

    def test_to_generated_preserves_name(self) -> None:
        """_to_generated() should carry the name through unchanged."""
        role = PredefinedUserRole(name=UserRole.ADMIN)
        generated = role._to_generated()
        assert generated.name == UserRole.ADMIN

    def test_accepts_all_user_roles(self) -> None:
        """PredefinedUserRole should work for every UserRole enum value."""
        for role_value in UserRole:
            role = PredefinedUserRole(name=role_value)
            generated = role._to_generated()
            assert generated.name == role_value


@pytest.mark.unit
class TestCustomUserRole:
    """Tests for the CustomUserRole convenience wrapper."""

    def test_to_generated_sets_custom_type(self) -> None:
        """_to_generated() should return a CustomUserRoleAssignment with type=custom."""
        role = CustomUserRole(id="role-abc-123")
        generated = role._to_generated()
        assert generated.type == "custom"

    def test_to_generated_preserves_id(self) -> None:
        """_to_generated() should carry the id through unchanged."""
        role = CustomUserRole(id="role-xyz-999")
        generated = role._to_generated()
        assert generated.id == "role-xyz-999"
