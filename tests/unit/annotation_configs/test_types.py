"""Tests for arize.annotation_configs.types public re-exports."""

from __future__ import annotations

import pytest

import arize.annotation_configs.types as types_module
from arize.annotation_configs.types import (
    AnnotationConfigsList200Response,
    AnnotationConfigType,
    CategoricalAnnotationValue,
    OptimizationDirection,
)


@pytest.mark.unit
class TestAnnotationConfigsTypes:
    """Tests for the annotation_configs types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "AnnotationConfigType" in types_module.__all__
        assert "AnnotationConfigsList200Response" in types_module.__all__
        assert "CategoricalAnnotationValue" in types_module.__all__
        assert "OptimizationDirection" in types_module.__all__

    def test_annotation_config_type_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(AnnotationConfigType, Enum)

    def test_optimization_direction_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(OptimizationDirection, Enum)

    def test_annotation_configs_list_response_is_class(self) -> None:
        assert isinstance(AnnotationConfigsList200Response, type)

    def test_categorical_annotation_value_is_class(self) -> None:
        assert isinstance(CategoricalAnnotationValue, type)
