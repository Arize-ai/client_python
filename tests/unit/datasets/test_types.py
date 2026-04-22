"""Tests for arize.datasets.types public re-exports."""

from __future__ import annotations

import pytest

import arize.datasets.types as types_module
from arize.datasets.types import (
    Dataset,
    DatasetsExamplesList200Response,
    DatasetsList200Response,
)


@pytest.mark.unit
class TestDatasetsTypes:
    """Tests for the datasets types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "Dataset" in types_module.__all__
        assert "DatasetsExamplesList200Response" in types_module.__all__
        assert "DatasetsList200Response" in types_module.__all__

    @pytest.mark.parametrize(
        "cls",
        [Dataset, DatasetsExamplesList200Response, DatasetsList200Response],
    )
    def test_type_is_class(self, cls: type) -> None:
        assert isinstance(cls, type)
