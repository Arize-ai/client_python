"""Tests for arize.resource_restrictions.types public re-exports."""

from __future__ import annotations

import pytest

import arize.resource_restrictions.types as types_module
from arize.resource_restrictions.types import ResourceRestriction


@pytest.mark.unit
class TestResourceRestrictionsTypes:
    """Tests for the resource_restrictions types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "ResourceRestriction" in types_module.__all__

    def test_resource_restriction_is_class(self) -> None:
        assert isinstance(ResourceRestriction, type)
