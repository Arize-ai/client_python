"""Tests for arize.organizations.types public re-exports."""

from __future__ import annotations

import pytest

import arize.organizations.types as types_module
from arize.organizations.types import Organization, OrganizationsList200Response


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
        assert "Organization" in types_module.__all__
        assert "OrganizationsList200Response" in types_module.__all__

    def test_organization_is_class(self) -> None:
        assert isinstance(Organization, type)

    def test_organizations_list_response_is_class(self) -> None:
        assert isinstance(OrganizationsList200Response, type)
