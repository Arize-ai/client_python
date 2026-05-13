"""Tests for arize.organizations.types public re-exports."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import arize.organizations.types as types_module
from arize.organizations.types import (
    CustomOrgRole,
    Organization,
    OrganizationMembership,
    OrganizationRole,
    OrganizationsList200Response,
    PredefinedOrgRole,
)


@pytest.mark.unit
class TestOrganizationsTypes:
    """Tests for the organizations types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "PredefinedOrgRole" in types_module.__all__
        assert "CustomOrgRole" in types_module.__all__
        assert "OrganizationMembership" in types_module.__all__
        assert "Organization" in types_module.__all__
        assert "OrganizationsList200Response" in types_module.__all__

    def test_organization_is_class(self) -> None:
        assert isinstance(Organization, type)

    def test_organizations_list_response_is_class(self) -> None:
        assert isinstance(OrganizationsList200Response, type)


@pytest.mark.unit
class TestPredefinedOrgRole:
    """Tests for the PredefinedOrgRole convenience wrapper."""

    def test_is_pydantic_model(self) -> None:
        from pydantic import BaseModel

        assert issubclass(PredefinedOrgRole, BaseModel)

    def test_type_field_defaults_to_predefined(self) -> None:
        role = PredefinedOrgRole(name=OrganizationRole.MEMBER)
        assert role.type == "predefined"

    def test_accepts_all_org_roles(self) -> None:
        """PredefinedOrgRole should accept every OrganizationRole enum value."""
        for role_value in OrganizationRole:
            role = PredefinedOrgRole(name=role_value)
            assert role.name == role_value

    def test_str_returns_role_value(self) -> None:
        role = PredefinedOrgRole(name=OrganizationRole.ADMIN)
        assert str(role) == OrganizationRole.ADMIN.value


@pytest.mark.unit
class TestCustomOrgRole:
    """Tests for the CustomOrgRole convenience wrapper."""

    def test_is_pydantic_model(self) -> None:
        from pydantic import BaseModel

        assert issubclass(CustomOrgRole, BaseModel)

    def test_type_field_defaults_to_custom(self) -> None:
        role = CustomOrgRole(id="role-abc-123")
        assert role.type == "custom"

    def test_name_defaults_to_none(self) -> None:
        role = CustomOrgRole(id="role-abc-123")
        assert role.name is None

    def test_name_preserved_when_set(self) -> None:
        role = CustomOrgRole(id="role-abc-123", name="My Custom Role")
        assert role.name == "My Custom Role"

    def test_str_returns_name_when_set(self) -> None:
        role = CustomOrgRole(id="role-abc-123", name="My Custom Role")
        assert str(role) == "My Custom Role"

    def test_str_returns_id_when_name_is_none(self) -> None:
        role = CustomOrgRole(id="role-abc-123")
        assert str(role) == "role-abc-123"


@pytest.mark.unit
class TestOrganizationMembership:
    """Tests for OrganizationMembership role coercion via field_validator."""

    def _make_role_assignment(self, actual_instance: object) -> MagicMock:
        from arize._generated.api_client.models.organization_role_assignment import (
            OrganizationRoleAssignment,
        )

        assignment = MagicMock(spec=OrganizationRoleAssignment)
        assignment.actual_instance = actual_instance
        return assignment

    def test_coerce_role_predefined(self) -> None:
        from arize._generated.api_client.models.organization_predefined_role_assignment import (
            OrganizationPredefinedRoleAssignment,
        )

        predefined = MagicMock(spec=OrganizationPredefinedRoleAssignment)
        predefined.name = OrganizationRole.ADMIN
        assignment = self._make_role_assignment(predefined)

        membership = OrganizationMembership(
            id="mem-1",
            user_id="user-1",
            organization_id="org-1",
            role=assignment,
        )

        assert membership.id == "mem-1"
        assert membership.user_id == "user-1"
        assert membership.organization_id == "org-1"
        assert isinstance(membership.role, PredefinedOrgRole)
        assert membership.role.name == OrganizationRole.ADMIN

    def test_coerce_role_custom(self) -> None:
        from arize._generated.api_client.models.organization_custom_role_assignment import (
            OrganizationCustomRoleAssignment,
        )

        custom = MagicMock(spec=OrganizationCustomRoleAssignment)
        custom.id = "custom-role-99"
        custom.name = "My Custom Role"
        assignment = self._make_role_assignment(custom)

        membership = OrganizationMembership(
            id="mem-1",
            user_id="user-1",
            organization_id="org-1",
            role=assignment,
        )

        assert isinstance(membership.role, CustomOrgRole)
        assert membership.role.id == "custom-role-99"
        assert membership.role.name == "My Custom Role"

    def test_coerce_role_unknown_raises(self) -> None:
        assignment = self._make_role_assignment(object())

        with pytest.raises(TypeError, match="Unknown org role type"):
            OrganizationMembership(
                id="mem-1",
                user_id="user-1",
                organization_id="org-1",
                role=assignment,
            )
