"""Tests for arize.role_bindings.types public re-exports."""

from __future__ import annotations

import pytest

import arize.role_bindings.types as types_module
from arize.role_bindings.types import RoleBinding, RoleBindingResourceType


@pytest.mark.unit
class TestRoleBindingsTypes:
    """Tests for the role_bindings types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "RoleBinding" in types_module.__all__
        assert "RoleBindingResourceType" in types_module.__all__

    def test_role_binding_resource_type_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(RoleBindingResourceType, Enum)

    def test_role_binding_is_class(self) -> None:
        assert isinstance(RoleBinding, type)
