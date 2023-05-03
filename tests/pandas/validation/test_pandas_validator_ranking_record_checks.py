from collections import ChainMap
from datetime import datetime, timedelta

import arize.pandas.validation.errors as err
import pandas as pd
import pytest
from arize.pandas.logger import Schema
from arize.pandas.validation.validator import Validator
from arize.utils.types import Environments, ModelTypes


def test_valid_record_ranking_prod():
    kwargs = get_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(["click", "favor", "click", None]),
                        "prediction_score": pd.Series([0.2, 0.3, float("NaN"), 0.5]),
                        "ranking_category": pd.Series(
                            [["click", "purchase"], None, ["favor"], ["click"]]
                        ),
                        "ranking_relevance": pd.Series([1, float("NaN"), 2, 0]),
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_record_ranking_pred_only_prod():
    kwargs = get_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(["click", "favor", "click", None]),
                        "prediction_score": pd.Series([0.2, 0.3, float("NaN"), 0.5]),
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_record_ranking_actual_only_prod():
    kwargs = get_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "ranking_category": pd.Series(
                            [["click", "purchase"], ["favor"], ["click"], None]
                        ),
                        "ranking_relevance": pd.Series([1, float("NaN"), 2, 0]),
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_record_ranking_prod():
    kwargs = get_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(["click", "favor", "click", None]),
                        "prediction_score": pd.Series([0.2, 0.3, 0.5, float("NaN")]),
                        "ranking_category": pd.Series(
                            [["click", "purchase"], ["favor"], ["click"], None]
                        ),
                        "ranking_relevance": pd.Series([1, 2, 0, float("NaN")]),
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == [
        "prediction_label",
        "prediction_score",
        "ranking_category",
        "ranking_relevance",
    ]
    assert errors[0].indexes == [3]


def test_invalid_record_ranking_pred_only_prod():
    kwargs = get_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(["click", "favor", "click", None]),
                        "prediction_score": pd.Series([0.2, 0.3, 0.5, float("NaN")]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["prediction_label", "prediction_score"]
    assert errors[0].indexes == [3]


def test_invalid_record_ranking_actual_only_prod():
    kwargs = get_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "ranking_category": pd.Series(
                            [["click", "purchase"], None, ["favor"], ["click"]]
                        ),
                        "ranking_relevance": pd.Series([1, float("NaN"), 2, 0]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["ranking_category", "ranking_relevance"]
    assert errors[0].indexes == [1]


def test_valid_record_ranking_training():
    kwargs = get_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(["click", "favor", "click", None]),
                        "prediction_score": pd.Series([0.2, 0.3, float("NaN"), 0.5]),
                        "ranking_category": pd.Series(
                            [["click", "purchase"], None, ["favor"], ["click"]]
                        ),
                        "ranking_relevance": pd.Series([1, 2, float("NaN"), 0]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_record_ranking_pred_only_training():
    kwargs = get_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(["click", "favor", "click", None]),
                        "prediction_score": pd.Series([0.2, 0.3, float("NaN"), 0.5]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_record_ranking_actual_only_training():
    kwargs = get_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "dataframe": pd.DataFrame(
                    {
                        "ranking_category": pd.Series(
                            [["click", "purchase"], ["favor"], ["click"], None]
                        ),
                        "ranking_relevance": pd.Series([1, float("NaN"), 2, 0]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_record_ranking_training():
    kwargs = get_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(["click", "favor", "click", None]),
                        "prediction_score": pd.Series([0.2, 0.3, 0.5, float("NaN")]),
                        "ranking_category": pd.Series(
                            [["click", "purchase"], ["favor"], ["click"], None]
                        ),
                        "ranking_relevance": pd.Series([1, 2, 0, float("NaN")]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 2
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["prediction_label", "prediction_score"]
    assert errors[0].indexes == [3]
    assert type(errors[1]) is err.InvalidRecord
    assert errors[1].columns == ["ranking_category", "ranking_relevance"]
    assert errors[1].indexes == [3]


def test_invalid_record_ranking_pred_only_training():
    kwargs = get_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(["click", "favor", "click", None]),
                        "prediction_score": pd.Series([0.2, 0.3, 0.5, float("NaN")]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["prediction_label", "prediction_score"]
    assert errors[0].indexes == [3]


def test_invalid_record_ranking_actual_only_training():
    kwargs = get_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "dataframe": pd.DataFrame(
                    {
                        "ranking_category": pd.Series(
                            [["click", "purchase"], None, ["favor"], ["click"]]
                        ),
                        "ranking_relevance": pd.Series([1, float("NaN"), 2, 0]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["ranking_category", "ranking_relevance"]
    assert errors[0].indexes == [1]


def get_kwargs():
    kwargs = {
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
            }
        ),
        "schema": Schema(
            prediction_id_column_name="prediction_id",
            prediction_group_id_column_name="prediction_group_id",
            timestamp_column_name="prediction_timestamp",
            feature_column_names=["item_type"],
            rank_column_name="ranking_rank",
            prediction_label_column_name="prediction_label",
            prediction_score_column_name="prediction_score",
            actual_score_column_name="ranking_relevance",
            actual_label_column_name="ranking_category",
        ),
    }
    return kwargs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
