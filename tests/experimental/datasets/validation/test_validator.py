import sys

import pytest

if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8 or higher", allow_module_level=True)

import pandas as pd
from arize.experimental.datasets.core.client import (
    ArizeDatasetsClient,
    _convert_default_columns_to_json_str,
)
from arize.experimental.datasets.validation.errors import (
    IDColumnUniqueConstraintError,
    RequiredColumnsError,
)
from arize.experimental.datasets.validation.validator import Validator


def test_happy_path():
    df = pd.DataFrame(
        {
            "user_data": [1, 2, 3],
        }
    )

    df_new = ArizeDatasetsClient._set_default_columns_for_dataset(df)
    differences = set(df_new.columns) ^ {"id", "created_at", "updated_at", "user_data"}
    assert not differences

    validation_errors = Validator.validate(df)
    assert len(validation_errors) == 0


def test_missing_columns():
    df = pd.DataFrame(
        {
            "user_data": [1, 2, 3],
        }
    )

    validation_errors = Validator.validate(df)
    assert len(validation_errors) == 1
    assert type(validation_errors[0]) is RequiredColumnsError


def test_non_unique_id_column():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "user_data": [1, 2, 3],
        }
    )
    df_new = ArizeDatasetsClient._set_default_columns_for_dataset(df)

    validation_errors = Validator.validate(df_new)
    assert len(validation_errors) == 1
    assert validation_errors[0] is IDColumnUniqueConstraintError


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires Python 3.8 or higher")
def test_dict_to_json_conversion() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "eval.MyEvaluator.metadata": [{"key": "value"}, {"key": "value"}, {"key": "value"}],
            "not_converted_dict_col": [{"key": "value"}, {"key": "value"}, {"key": "value"}],
        }
    )
    # before conversion, the column with the evaluator name is a dict
    assert type(df["eval.MyEvaluator.metadata"][0]) is dict
    assert type(df["not_converted_dict_col"][0]) is dict

    # Check that only the column with the evaluator name is converted to JSON
    converted_df = _convert_default_columns_to_json_str(df)
    assert type(converted_df["eval.MyEvaluator.metadata"][0]) is str
    assert type(converted_df["not_converted_dict_col"][0]) is dict
