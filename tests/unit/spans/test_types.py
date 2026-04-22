"""Tests for arize.spans.types public re-exports."""

from __future__ import annotations

import pytest

import arize.spans.types as types_module
from arize.spans.types import SpansList200Response


@pytest.mark.unit
class TestSpansTypes:
    """Tests for the spans types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "SpansList200Response" in types_module.__all__

    def test_spans_list_response_is_class(self) -> None:
        assert isinstance(SpansList200Response, type)
