import sys

import pytest

if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8 or higher", allow_module_level=True)

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from arize.experimental.datasets.core.client import (
    ArizeDatasetsClient,
    _convert_datetime_columns_to_int,
    _convert_default_columns_to_json_str,
)
from arize.experimental.datasets.validation.errors import (
    EmptyDatasetError,
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
    differences = set(df_new.columns) ^ {
        "id",
        "created_at",
        "updated_at",
        "user_data",
    }
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


@pytest.mark.skipif(
    sys.version_info < (3, 8), reason="Requires Python 3.8 or higher"
)
def test_dict_to_json_conversion() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "eval.MyEvaluator.metadata": [
                {"key": "value"},
                {"key": "value"},
                {"key": "value"},
            ],
            "not_converted_dict_col": [
                {"key": "value"},
                {"key": "value"},
                {"key": "value"},
            ],
        }
    )
    # before conversion, the column with the evaluator name is a dict
    assert type(df["eval.MyEvaluator.metadata"][0]) is dict
    assert type(df["not_converted_dict_col"][0]) is dict

    # Check that only the column with the evaluator name is converted to JSON
    converted_df = _convert_default_columns_to_json_str(df)
    assert type(converted_df["eval.MyEvaluator.metadata"][0]) is str
    assert type(converted_df["not_converted_dict_col"][0]) is dict


def test_datetime_conversion():
    """Test the datetime conversion function with various datetime types"""

    # Create a test DataFrame with various datetime columns
    test_data = {
        "id": [1, 2, 3],
        "text": ["hello", "world", "test"],
        "created_at": [
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 3, 12, 0, 0, tzinfo=timezone.utc),
        ],
        "updated_at": [
            pd.Timestamp("2024-01-01T12:00:00Z"),
            pd.Timestamp("2024-01-02T12:00:00Z"),
            pd.Timestamp("2024-01-03T12:00:00Z"),
        ],
        "event_time": [
            np.datetime64("2024-01-01T12:00:00.000000000"),
            np.datetime64("2024-01-02T12:00:00.000000000"),
            np.datetime64("2024-01-03T12:00:00.000000000"),
        ],
        "regular_col": [1, 2, 3],
    }

    df = pd.DataFrame(test_data)

    # Verify original data types
    assert "datetime64" in str(df["created_at"].dtype)
    assert "datetime64" in str(df["updated_at"].dtype)
    assert "datetime64" in str(df["event_time"].dtype)
    assert df["regular_col"].dtype == "int64"

    # Apply the conversion
    converted_df = _convert_datetime_columns_to_int(df.copy())

    # Verify converted data types
    assert converted_df["created_at"].dtype == "int64"
    assert converted_df["updated_at"].dtype == "int64"
    assert converted_df["event_time"].dtype == "int64"
    assert converted_df["regular_col"].dtype == "int64"  # Should be unchanged

    # Verify the conversion values
    for col in ["created_at", "updated_at", "event_time"]:
        for i, val in enumerate(converted_df[col]):
            original = df[col].iloc[i]
            expected_ms = int(original.timestamp() * 1000)
            assert (
                val == expected_ms
            ), f"Conversion failed for {col} row {i}: got {val}, expected {expected_ms}"

    # Verify non-datetime columns are unchanged
    assert converted_df["regular_col"].equals(df["regular_col"])
    assert converted_df["text"].equals(df["text"])


def test_datetime_conversion_with_no_datetime_columns():
    """Test that conversion works correctly when no datetime columns are present"""

    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": ["hello", "world", "test"],
            "number": [1.5, 2.5, 3.5],
        }
    )

    converted_df = _convert_datetime_columns_to_int(df.copy())

    # Should be identical since no datetime columns
    pd.testing.assert_frame_equal(df, converted_df)


def test_datetime_conversion_with_mixed_types():
    """Test conversion with mixed datetime and non-datetime columns"""

    df = pd.DataFrame(
        {
            "id": [1, 2],
            "timestamp": [
                pd.Timestamp("2024-01-01T12:00:00Z"),
                pd.Timestamp("2024-01-02T12:00:00Z"),
            ],
            "string_col": ["a", "b"],
            "int_col": [10, 20],
        }
    )

    converted_df = _convert_datetime_columns_to_int(df.copy())

    # Check that only datetime columns were converted
    assert converted_df["timestamp"].dtype == "int64"
    assert converted_df["string_col"].dtype == "object"  # Should remain string
    assert converted_df["int_col"].dtype == "int64"  # Should remain int

    # Verify conversion values
    for i, val in enumerate(converted_df["timestamp"]):
        original = df["timestamp"].iloc[i]
        expected_ms = int(original.timestamp() * 1000)
        assert val == expected_ms


def test_empty_dataset():
    df = pd.DataFrame(columns=["id", "user_data"])
    df_new = ArizeDatasetsClient._set_default_columns_for_dataset(df)

    validation_errors = Validator.validate(df_new)
    assert len(validation_errors) == 1
    assert validation_errors[0] is EmptyDatasetError
