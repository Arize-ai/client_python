import uuid
from collections import ChainMap
from datetime import datetime, timedelta

import arize.pandas.validation.errors as err
import pandas as pd
import pytest
from arize.pandas.logger import Schema
from arize.pandas.validation.validator import Validator
from arize.utils.constants import MAX_PREDICTION_ID_LEN
from arize.utils.types import Environments, ModelTypes

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
            "prediction_id": pd.Series([str(uuid.uuid4()) for _ in range(4)]),
            "prediction_group_id": pd.Series(["X" * 10, "X" * 10, "Y" * 10, "Y" * 10]),
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


def test_ranking_values_happy_path():
    errors = Validator.validate_values(**kwargs)
    assert len(errors) == 0


def test_rank_is_not_null_and_between_1_and_100():
    ranks_good = pd.Series([1, 5, 10, 100])
    ranks_out_of_lower_bound = pd.Series([0, 5, 10, 100])
    ranks_out_of_upper_bound = pd.Series([1, 5, 10, 101])
    ranks_with_none = pd.Series([1, 5, 10, None])

    # happy path
    errors = Validator.validate_values(
        **ChainMap(
            {"dataframe": pd.DataFrame({"ranking_rank": ranks_good})},
            kwargs,
        )
    )
    assert len(errors) == 0

    # missing value
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "ranking_rank": ranks_with_none,
                        "ranking_category": pd.Series(
                            [["click", "purchase"], ["click", "favor"], ["favor"], ["click"]]
                        ),
                        "ranking_relevance": pd.Series([1, 0, 2, 0]),
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue

    for ranks in (ranks_out_of_upper_bound, ranks_out_of_lower_bound):
        errors = Validator.validate_values(
            **ChainMap({"dataframe": pd.DataFrame({"ranking_rank": ranks})}, kwargs)
        )
        assert len(errors) == 1
        assert type(errors[0]) is err.InvalidRankValue


def test_prediction_group_id_length():
    null_ids = pd.Series([None, None, "A" * 10, "B" * 10])
    empty_ids = pd.Series(["", "", "", ""])
    long_ids = pd.Series(["A" * (MAX_PREDICTION_ID_LEN + 1)] * 4)

    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_group_id": null_ids,
                        "ranking_category": pd.Series(
                            [["click", "purchase"], ["click", "favor"], ["favor"], ["click"]]
                        ),
                        "ranking_relevance": pd.Series([1, 0, 2, 0]),
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue

    for ids in (empty_ids, long_ids):
        errors = Validator.validate_values(
            **ChainMap({"dataframe": pd.DataFrame({"prediction_group_id": ids})}, kwargs)
        )
        assert len(errors) == 1
        assert type(errors[0]) is err.InvalidStringLengthInColumn


def test_ranking_category_empty_list():
    null_category = pd.Series([None, None, ["buy", "click"], ["buy"]])
    empty_list_category = pd.Series([[], [], ["buy", "click"], ["buy"]])

    # null values are ok for ranking actual_label_column
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "ranking_category": null_category,
                        "ranking_relevance": pd.Series([1, 0, 2, 0]),
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0

    # empty lists are NOT ok for ranking actual_label_column
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "ranking_category": empty_list_category,
                        "ranking_relevance": pd.Series([1, 0, 2, 0]),
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
