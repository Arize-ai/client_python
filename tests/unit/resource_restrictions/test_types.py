"""Tests for arize.resource_restrictions.types public re-exports."""

from __future__ import annotations

import pytest

import arize  # noqa: F401 — triggers monkey-patching
import arize.resource_restrictions.types as types_module
from arize._generated.api_client.models.pagination_metadata import (
    PaginationMetadata,
)
from arize.resource_restrictions.types import (
    ResourceRestriction,
    ResourceRestrictionListResponse,
    ResourceRestrictionType,
)


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
        assert "ResourceRestrictionListResponse" in types_module.__all__
        assert "ResourceRestrictionType" in types_module.__all__

    def test_resource_restriction_is_class(self) -> None:
        assert isinstance(ResourceRestriction, type)

    def test_resource_restriction_list_response_is_class(self) -> None:
        assert isinstance(ResourceRestrictionListResponse, type)

    def test_resource_restriction_type_has_project(self) -> None:
        assert ResourceRestrictionType.PROJECT.value == "PROJECT"

    def test_resource_restriction_list_response_has_to_df(self) -> None:
        assert hasattr(ResourceRestrictionListResponse, "to_df")

    def test_resource_restriction_list_response_to_df_empty(self) -> None:
        response = ResourceRestrictionListResponse(
            resource_restrictions=[],
            pagination=PaginationMetadata(has_more=False, next_cursor=None),
        )
        df = response.to_df()
        assert df.empty
