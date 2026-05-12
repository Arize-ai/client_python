"""Tests for arize.spaces.types public re-exports."""

from __future__ import annotations

import pytest

import arize.spaces.types as types_module
from arize.spaces.types import (
    PredefinedSpaceRole,
    Space,
    SpacesList200Response,
    UserSpaceRole,
)


@pytest.mark.unit
class TestSpacesTypes:
    """Tests for the spaces types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "PredefinedSpaceRole" in types_module.__all__
        assert "Space" in types_module.__all__
        assert "SpacesList200Response" in types_module.__all__

    def test_space_is_class(self) -> None:
        assert isinstance(Space, type)

    def test_spaces_list_response_is_class(self) -> None:
        assert isinstance(SpacesList200Response, type)


@pytest.mark.unit
class TestPredefinedSpaceRole:
    """Tests for the PredefinedSpaceRole convenience wrapper."""

    def test_to_generated_sets_predefined_type(self) -> None:
        """_to_generated() should return a PredefinedRoleAssignment with type=predefined."""
        role = PredefinedSpaceRole(name=UserSpaceRole.MEMBER)
        generated = role._to_generated()
        assert generated.type == "predefined"

    def test_to_generated_preserves_name(self) -> None:
        """_to_generated() should carry the name through unchanged."""
        role = PredefinedSpaceRole(name=UserSpaceRole.ADMIN)
        generated = role._to_generated()
        assert generated.name == UserSpaceRole.ADMIN

    def test_accepts_all_space_roles(self) -> None:
        """PredefinedSpaceRole should work for every UserSpaceRole enum value."""
        for role_value in UserSpaceRole:
            role = PredefinedSpaceRole(name=role_value)
            generated = role._to_generated()
            assert generated.name == role_value
