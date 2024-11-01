from collections import ChainMap

import pandas as pd
import pytest

from arize.pandas.logger import Schema
from arize.pandas.validation.errors import InvalidShapSuffix
from arize.pandas.validation.validator import Validator
from arize.utils.types import Environments, ModelTypes


def test_invalid_feature_columns():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": kwargs["dataframe"].assign(feat_shap=[0]),
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    feature_column_names=["feat_shap"],
                ),
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is InvalidShapSuffix


def test_invalid_tag_columns():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": kwargs["dataframe"].assign(tag_shap=[0]),
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    tag_column_names=["tag_shap"],
                ),
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is InvalidShapSuffix


def test_invalid_shap_columns():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": kwargs["dataframe"].assign(shap=[0]),
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    shap_values_column_names={"feat_shap": "shap"},
                ),
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is InvalidShapSuffix


def test_invalid_multiple():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": kwargs["dataframe"].assign(
                    feat_shap=[0], tag_shap=[0], shap=[0]
                ),
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    feature_column_names=["feat_shap"],
                    tag_column_names=["tag_shap"],
                    shap_values_column_names={"feat_shap": "shap"},
                ),
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert type(errors[0]) is InvalidShapSuffix


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
