"""Tests for arize.spaces.types public re-exports."""

from __future__ import annotations

import pytest

import arize.spaces.types as types_module
from arize.spaces.types import Space, SpacesList200Response


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
        assert "Space" in types_module.__all__
        assert "SpacesList200Response" in types_module.__all__

    def test_space_is_class(self) -> None:
        assert isinstance(Space, type)

    def test_spaces_list_response_is_class(self) -> None:
        assert isinstance(SpacesList200Response, type)
