"""Unit tests for src/arize/ml/batch_validation/validator.py."""

from __future__ import annotations

import pandas as pd
import pytest

from arize.ml.batch_validation.errors import (
    MissingProductionPredActFeatureImportance,
)
from arize.ml.batch_validation.validator import Validator
from arize.ml.types import Environments, ModelTypes, Schema


def _errors(schema: Schema, environment: Environments) -> list:
    """Run validate_params for a simple dataframe and return the error list."""
    df = pd.DataFrame(
        {
            "prediction_id": ["a", "b", "c"],
            "feature_x": [1.0, 2.0, 3.0],
            "prediction_label": ["1", "0", "1"],
            "actual_label": ["1", "0", "0"],
        }
    )
    return Validator.validate_params(
        dataframe=df,
        model_id="m",
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=environment,
        schema=schema,
        model_version="1.0",
    )


@pytest.mark.unit
class TestProductionFeaturesOnlyRejection:
    """A features-only PRODUCTION schema can never create a model (issue #80409)."""

    def test_features_only_production_is_rejected(self) -> None:
        """Features but no prediction/actual/FI columns must raise the new error."""
        schema = Schema(
            prediction_id_column_name="prediction_id",
            feature_column_names=["feature_x"],
        )
        errors = _errors(schema, Environments.PRODUCTION)
        assert any(
            isinstance(e, MissingProductionPredActFeatureImportance)
            for e in errors
        ), f"expected rejection, got: {[repr(e) for e in errors]}"

    def test_production_with_prediction_is_allowed(self) -> None:
        """A normal prediction payload must NOT trip the new error."""
        schema = Schema(
            prediction_id_column_name="prediction_id",
            feature_column_names=["feature_x"],
            prediction_label_column_name="prediction_label",
        )
        errors = _errors(schema, Environments.PRODUCTION)
        assert not any(
            isinstance(e, MissingProductionPredActFeatureImportance)
            for e in errors
        )

    def test_delayed_actuals_production_is_allowed(self) -> None:
        """Delayed actuals (actual columns, no prediction) is a legit workflow."""
        schema = Schema(
            prediction_id_column_name="prediction_id",
            actual_label_column_name="actual_label",
        )
        errors = _errors(schema, Environments.PRODUCTION)
        assert not any(
            isinstance(e, MissingProductionPredActFeatureImportance)
            for e in errors
        )
        assert schema.is_delayed() is True

    def test_features_only_training_uses_preprod_guard_not_prod(self) -> None:
        """Preprod features-only is handled by the existing preprod guard, not this one."""
        schema = Schema(
            prediction_id_column_name="prediction_id",
            feature_column_names=["feature_x"],
        )
        errors = _errors(schema, Environments.TRAINING)
        assert not any(
            isinstance(e, MissingProductionPredActFeatureImportance)
            for e in errors
        )
