"""Unit tests for src/arize/spans/conversion.py."""

from __future__ import annotations

import numpy as np
import pytest

from arize.spans.conversion import is_missing_value


@pytest.mark.unit
class TestIsMissingValue:
    def test_empty_list_is_not_missing(self) -> None:
        """Empty list must not raise and must return False (regression for GH#kiko/fix-an-bug)."""
        assert is_missing_value([]) is False

    def test_nonempty_list_is_not_missing(self) -> None:
        assert is_missing_value([{"role": "user", "content": "hi"}]) is False

    def test_none_is_missing(self) -> None:
        assert is_missing_value(None) is True

    def test_nan_is_missing(self) -> None:
        assert is_missing_value(float("nan")) is True

    def test_inf_is_missing(self) -> None:
        assert is_missing_value(np.inf) is True
        assert is_missing_value(-np.inf) is True

    def test_scalar_string_is_not_missing(self) -> None:
        assert is_missing_value("hello") is False

    def test_empty_ndarray_is_not_missing(self) -> None:
        assert is_missing_value(np.array([])) is False

    def test_nonempty_ndarray_is_not_missing(self) -> None:
        assert is_missing_value(np.array([1, 2, 3])) is False
