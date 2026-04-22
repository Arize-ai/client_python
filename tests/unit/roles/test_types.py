"""Tests for arize.roles.types public re-exports."""

from __future__ import annotations

import pytest

import arize.roles.types as types_module
from arize.roles.types import Permission, Role, RolesList200Response


@pytest.mark.unit
class TestRolesTypes:
    """Tests for the roles types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "Permission" in types_module.__all__
        assert "Role" in types_module.__all__
        assert "RolesList200Response" in types_module.__all__

    def test_permission_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(Permission, Enum)

    def test_role_is_class(self) -> None:
        assert isinstance(Role, type)

    def test_roles_list_response_is_class(self) -> None:
        assert isinstance(RolesList200Response, type)
