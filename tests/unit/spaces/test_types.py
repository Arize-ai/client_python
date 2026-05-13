"""Tests for arize.spaces.types public re-exports."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import arize.spaces.types as types_module
from arize.spaces.types import (
    CustomSpaceRole,
    PredefinedSpaceRole,
    Space,
    SpaceMembership,
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
        assert "CustomSpaceRole" in types_module.__all__
        assert "SpaceMembership" in types_module.__all__
        assert "Space" in types_module.__all__
        assert "SpacesList200Response" in types_module.__all__

    def test_space_is_class(self) -> None:
        assert isinstance(Space, type)

    def test_spaces_list_response_is_class(self) -> None:
        assert isinstance(SpacesList200Response, type)


@pytest.mark.unit
class TestPredefinedSpaceRole:
    """Tests for the PredefinedSpaceRole convenience wrapper."""

    def test_is_pydantic_model(self) -> None:
        from pydantic import BaseModel

        assert issubclass(PredefinedSpaceRole, BaseModel)

    def test_type_field_defaults_to_predefined(self) -> None:
        role = PredefinedSpaceRole(name=UserSpaceRole.MEMBER)
        assert role.type == "predefined"

    def test_accepts_all_space_roles(self) -> None:
        """PredefinedSpaceRole should accept every UserSpaceRole enum value."""
        for role_value in UserSpaceRole:
            role = PredefinedSpaceRole(name=role_value)
            assert role.name == role_value


@pytest.mark.unit
class TestCustomSpaceRole:
    """Tests for the CustomSpaceRole convenience wrapper."""

    def test_is_pydantic_model(self) -> None:
        from pydantic import BaseModel

        assert issubclass(CustomSpaceRole, BaseModel)

    def test_type_field_defaults_to_custom(self) -> None:
        role = CustomSpaceRole(id="role-xyz-42")
        assert role.type == "custom"

    def test_name_defaults_to_none(self) -> None:
        role = CustomSpaceRole(id="role-xyz-42")
        assert role.name is None

    def test_name_preserved_when_set(self) -> None:
        role = CustomSpaceRole(id="role-xyz-42", name="Space Viewer")
        assert role.name == "Space Viewer"


@pytest.mark.unit
class TestSpaceMembership:
    """Tests for SpaceMembership role coercion via field_validator."""

    def _make_role_assignment(self, actual_instance: object) -> MagicMock:
        from arize._generated.api_client.models.space_role_assignment import (
            SpaceRoleAssignment,
        )

        assignment = MagicMock(spec=SpaceRoleAssignment)
        assignment.actual_instance = actual_instance
        return assignment

    def test_coerce_role_predefined(self) -> None:
        from arize._generated.api_client.models.predefined_role_assignment import (
            PredefinedRoleAssignment,
        )

        predefined = MagicMock(spec=PredefinedRoleAssignment)
        predefined.name = UserSpaceRole.MEMBER
        assignment = self._make_role_assignment(predefined)

        membership = SpaceMembership(
            id="mem-2", user_id="user-2", space_id="space-2", role=assignment
        )

        assert membership.id == "mem-2"
        assert membership.user_id == "user-2"
        assert membership.space_id == "space-2"
        assert isinstance(membership.role, PredefinedSpaceRole)
        assert membership.role.name == UserSpaceRole.MEMBER

    def test_coerce_role_custom(self) -> None:
        from arize._generated.api_client.models.custom_role_assignment import (
            CustomRoleAssignment,
        )

        custom = MagicMock(spec=CustomRoleAssignment)
        custom.id = "custom-space-role-7"
        custom.name = "Space Operator"
        assignment = self._make_role_assignment(custom)

        membership = SpaceMembership(
            id="mem-2", user_id="user-2", space_id="space-2", role=assignment
        )

        assert isinstance(membership.role, CustomSpaceRole)
        assert membership.role.id == "custom-space-role-7"
        assert membership.role.name == "Space Operator"

    def test_coerce_role_unknown_raises(self) -> None:
        assignment = self._make_role_assignment(object())

        with pytest.raises(TypeError, match="Unknown space role type"):
            SpaceMembership(
                id="mem-2",
                user_id="user-2",
                space_id="space-2",
                role=assignment,
            )
