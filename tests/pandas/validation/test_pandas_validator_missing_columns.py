import pytest
import pandas as pd
from collections import ChainMap

from arize.pandas.logger import Schema
from arize.utils.types import Environments, ModelTypes

from arize.pandas.validation.validator import Validator
from arize.pandas.validation.errors import MissingColumns


def test_missing_prediction_id():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": pd.DataFrame({"prediction_label": pd.Series(["fraud"])}),
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                ),
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingColumns


def test_missing_timestamp():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    timestamp_column_name="prediction_timestamp",
                )
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingColumns


def test_missing_feature_columns():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    feature_column_names=["A"],
                )
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingColumns


def test_missing_shap_columns():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    shap_values_column_names={"A": "aa"},
                )
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingColumns


def test_missing_prediction_label():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="B",
                )
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingColumns


def test_missing_prediction_score():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    prediction_score_column_name="C",
                )
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingColumns


def test_missing_actual_label():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    actual_label_column_name="D",
                )
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingColumns


def test_missing_actual_score():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    actual_score_column_name="E",
                )
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingColumns


def test_missing_multiple():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    timestamp_column_name="prediction_timestamp",
                    feature_column_names=["A"],
                    shap_values_column_names={"A": "aa"},
                    prediction_label_column_name="B",
                    prediction_score_column_name="C",
                    actual_label_column_name="D",
                    actual_score_column_name="E",
                )
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is MissingColumns


kwargs = {
    "model_id": "fraud",
    "model_version": "v1.0",
    "model_type": ModelTypes.SCORE_CATEGORICAL,
    "environment": Environments.PRODUCTION,
    "dataframe": pd.DataFrame(
        {
            "prediction_id": pd.Series(["0"]),
            "prediction_label": pd.Series(["fraud"]),
            "prediction_score": pd.Series([1]),
            "actual_label": pd.Series(["not fraud"]),
            "actual_score": pd.Series([0]),
        }
    ),
    "schema": Schema(
        prediction_id_column_name="prediction_id",
        prediction_label_column_name="prediction_label",
    ),
}

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
