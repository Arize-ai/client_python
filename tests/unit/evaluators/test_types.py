"""Tests for arize.evaluators.types public re-exports."""

from __future__ import annotations

import pytest

import arize.evaluators.types as types_module
from arize.evaluators.types import (
    Evaluator,
    EvaluatorLlmConfig,
    EvaluatorsList200Response,
    EvaluatorVersion,
    EvaluatorVersionsList200Response,
    EvaluatorWithVersion,
    TemplateConfig,
)


@pytest.mark.unit
class TestEvaluatorsTypes:
    """Tests for the evaluators types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        expected = {
            "Evaluator",
            "EvaluatorLlmConfig",
            "EvaluatorVersion",
            "EvaluatorVersionsList200Response",
            "EvaluatorWithVersion",
            "EvaluatorsList200Response",
            "TemplateConfig",
        }
        assert expected.issubset(set(types_module.__all__))

    @pytest.mark.parametrize(
        "cls",
        [
            Evaluator,
            EvaluatorLlmConfig,
            EvaluatorVersion,
            EvaluatorVersionsList200Response,
            EvaluatorWithVersion,
            EvaluatorsList200Response,
            TemplateConfig,
        ],
    )
    def test_type_is_class(self, cls: type) -> None:
        assert isinstance(cls, type)
