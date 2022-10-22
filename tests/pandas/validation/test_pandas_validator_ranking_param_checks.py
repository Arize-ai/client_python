from collections import ChainMap
from datetime import datetime, timedelta

import pandas as pd
import pytest

import arize.pandas.validation.errors as err
from arize.pandas.logger import Schema
from arize.pandas.validation.validator import Validator
from arize.utils.types import Environments, ModelTypes

kwargs = {
    "model_id": "rank",
    "model_type": ModelTypes.RANKING,
    "environment": Environments.PRODUCTION,
    "dataframe": pd.DataFrame(
        {
            "prediction_timestamp": pd.Series(
                [
                    datetime.now(),
                    datetime.now() + timedelta(days=1),
                    datetime.now() - timedelta(days=364),
                    datetime.now() + timedelta(days=364),
                ]
            ),
            "prediction_id": pd.Series(["x_1", "x_2", "y_1", "y_2"]),
            "prediction_group_id": pd.Series(["X", "X", "Y", "Y"]),
            "item_type": pd.Series(["toy", "game", "game", "pens"]),
            "ranking_rank": pd.Series([1, 2, 1, 2]),
            "ranking_category": pd.Series(
                [["click", "purchase"], ["click", "favor"], ["favor"], ["click"]]
            ),
            "ranking_relevance": pd.Series([1, 0, 2, 0]),
        }
    ),
    "schema": Schema(
        prediction_id_column_name="prediction_id",
        prediction_group_id_column_name="prediction_group_id",
        timestamp_column_name="prediction_timestamp",
        feature_column_names=["item_type"],
        rank_column_name="ranking_rank",
        actual_score_column_name="ranking_relevance",
        actual_label_column_name="ranking_category",
    ),
}


def test_ranking_param_check_happy_path():
    errors = Validator.validate_params(**kwargs)
    assert len(errors) == 0


def test_ranking_param_check_missing_prediction_group_id():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_group_id_column_name=None,
                    timestamp_column_name="prediction_timestamp",
                    feature_column_names=["item_type"],
                    rank_column_name="ranking_rank",
                    actual_score_column_name="ranking_relevance",
                    actual_label_column_name="ranking_category",
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingRequiredColumnsForRankingModel


def test_ranking_param_check_missing_rank():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_group_id_column_name="prediction_group_id",
                    timestamp_column_name="prediction_timestamp",
                    feature_column_names=["item_type"],
                    rank_column_name=None,
                    actual_score_column_name="ranking_relevance",
                    actual_label_column_name="ranking_category",
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingRequiredColumnsForRankingModel


def test_ranking_param_check_missing_category():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_group_id_column_name="prediction_group_id",
                    timestamp_column_name="prediction_timestamp",
                    feature_column_names=["item_type"],
                    rank_column_name="ranking_rank",
                    actual_score_column_name=None,
                    actual_label_column_name=None,
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
