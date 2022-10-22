from collections import ChainMap
from datetime import datetime, timedelta

import arize.pandas.validation.errors as err
import pandas as pd
import pyarrow as pa
import pytest
from arize.pandas.logger import Schema
from arize.pandas.validation.validator import Validator
from arize.utils.types import ModelTypes

kwargs = {
    "model_type": ModelTypes.RANKING,
    "pyarrow_schema": pa.Schema.from_pandas(
        pd.DataFrame(
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
                "ranking_relevance": pd.Series([1.0, 2.0, 2.5, 0.1]),
            }
        )
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


def test_ranking_columns_type_check_happy_path_1():
    errors = Validator.validate_types(**kwargs)
    assert len(errors) == 0


def test_ranking_columns_type_check_happy_path_2():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {
                            "prediction_group_id": pd.Series([1, 2, 3, 4]),
                        }
                    )
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_type_prediction_group_id():
    invalid_typed_group_id = (
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        pd.Series([["1.0"], ["2.0"], ["3.0"], ["4.0"]]),
        pd.Series([datetime.now()] * 4),
        pd.Series([True, False, True, False]),
    )
    for ids in invalid_typed_group_id:
        errors = Validator.validate_types(
            **ChainMap(
                {
                    "pyarrow_schema": pa.Schema.from_pandas(
                        pd.DataFrame({"prediction_group_id": ids})
                    ),
                },
                kwargs,
            )
        )
        assert len(errors) == 1
        assert type(errors[0]) is err.InvalidType


def test_invalid_type_rank():
    invalid_typed_rank = (
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        pd.Series([["1.0"], ["2.0"], ["3.0"], ["4.0"]]),
        pd.Series([datetime.now()] * 4),
        pd.Series([True, False, True, False]),
    )

    for ranks in invalid_typed_rank:
        errors = Validator.validate_types(
            **ChainMap(
                {
                    "pyarrow_schema": pa.Schema.from_pandas(
                        pd.DataFrame(
                            {
                                "ranking_rank": ranks,
                            }
                        )
                    ),
                },
                kwargs,
            )
        )
        assert len(errors) == 1
        assert type(errors[0]) is err.InvalidType


def test_invalid_type_rank_category():
    invalid_typed_category = (
        pd.Series([1, 2, 3, 4]),
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        pd.Series([True, False, True, False]),
        pd.Series([datetime.now()] * 4),
        pd.Series([[1], [2, 3, 4], [5, 6], [7, 8, 9]]),
    )
    for catetory in invalid_typed_category:
        errors = Validator.validate_types(
            **ChainMap(
                {
                    "pyarrow_schema": pa.Schema.from_pandas(
                        pd.DataFrame(
                            {
                                "ranking_category": catetory,
                            }
                        )
                    ),
                },
                kwargs,
            )
        )
        assert len(errors) == 1
        assert type(errors[0]) is err.InvalidType


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
