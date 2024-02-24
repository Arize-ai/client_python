import sys
from dataclasses import dataclass
from typing import Optional, Set

import numpy as np
import pandas as pd
import pytest
from arize.pandas.etl.casting import (
    ColumnCastingError,
    InvalidTypedColumnsError,
    cast_typed_columns,
)
from arize.utils.types import Schema, TypedColumns


@dataclass
class CastingTestCase:
    name: str
    df: pd.DataFrame
    features: Optional[TypedColumns]
    tags: Optional[TypedColumns]
    expected: dict

    def get_all_inferred_columns(self) -> Set[str]:
        all_columns = set()
        typed_columns = set()
        for field in [self.features, self.tags]:
            if isinstance(field, TypedColumns):
                all_columns.update(set(field.get_all_column_names()))
                typed_columns.update(
                    set(
                        sum(
                            (
                                field.to_str or [],
                                field.to_int or [],
                                field.to_float or [],
                            ),
                            [],
                        )
                    )
                )
        return all_columns - typed_columns


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_duplicates_error():
    df = pd.DataFrame({"A": pd.Series([int(1), int(2)])})
    features = TypedColumns(
        to_int=["A"],
        inferred=["B", "A"],
    )
    tags = TypedColumns(
        to_int=["A"],
        inferred=["B", "B"],
    )
    schema = get_schema(features, tags)
    with pytest.raises(InvalidTypedColumnsError) as excinfo:
        _, _ = cast_typed_columns(df, schema)
    # The casting order isn't deterministic; the error message could be either of the following:
    reasons = {"has duplicate column names: A", "has duplicate column names: B"}
    assert excinfo.value.reason in reasons


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_empty_error():
    df = pd.DataFrame({"A": pd.Series([int(1), int(2)])})
    features = TypedColumns()
    tags = TypedColumns()
    schema = get_schema(features, tags)
    with pytest.raises(InvalidTypedColumnsError) as excinfo:
        _, _ = cast_typed_columns(df, schema)
    assert excinfo.value.reason == "is empty"

    features = None
    tags = TypedColumns(
        inferred=[],
        to_int=[],
        to_float=[],
        to_str=[],
    )
    schema = get_schema(features, tags)
    with pytest.raises(InvalidTypedColumnsError) as excinfo:
        _, _ = cast_typed_columns(df, schema)
    assert excinfo.value.reason == "is empty"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_cast_string_to_float_error():
    df = pd.DataFrame({"A": pd.Series([int(1), str("hello")])})
    features = TypedColumns(
        to_float=["A"],
    )
    tags = ["A"]
    schema = get_schema(features, tags)
    with pytest.raises(ColumnCastingError) as excinfo:
        _, _ = cast_typed_columns(df, schema)
    assert excinfo.value.error_msg == "could not convert string to float: 'hello'"
    assert excinfo.value.attempted_casting_columns == ["A"]
    assert excinfo.value.attempted_casting_type == "Float64"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_cast_numeric_string_to_int_error():
    df = pd.DataFrame(
        {
            "A": pd.Series([int(1), str("2.5")]),
            "B": pd.Series([int(1), str("2.5")]),
            "C": pd.Series([int(1), str("2.5")]),
        }
    )
    features = ["A"]
    tags = TypedColumns(
        to_int=["A", "B", "C"],
    )
    schema = get_schema(features, tags)
    with pytest.raises(ColumnCastingError) as excinfo:
        _, _ = cast_typed_columns(df, schema)
    assert excinfo.value.error_msg == "invalid literal for int() with base 10: '2.5'"
    assert excinfo.value.attempted_casting_columns == ["A", "B", "C"]
    assert excinfo.value.attempted_casting_type == "Int64"
    assert (
        excinfo.value.error_message() == "Failed to cast to type Int64 for columns: A, B and C. "
        "Error: invalid literal for int() with base 10: '2.5'"
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_cast_empty_string_to_numeric_error():
    df = pd.DataFrame({"A": pd.Series([int(1), str("")])})
    features = TypedColumns(
        to_int=["A"],
    )
    tags = TypedColumns(
        to_str=["A"],
    )
    schema = get_schema(features, tags)
    with pytest.raises(ColumnCastingError) as excinfo:
        _, _ = cast_typed_columns(df, schema)
    assert excinfo.value.error_msg == "invalid literal for int() with base 10: ''"
    assert excinfo.value.attempted_casting_columns == ["A"]
    assert excinfo.value.attempted_casting_type == "Int64"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_cast_float_to_int_error():
    df = pd.DataFrame({"A": pd.Series([int(1), float(2.5)])})
    features = ["A"]
    tags = TypedColumns(
        to_int=["A"],
    )
    schema = get_schema(features, tags)
    with pytest.raises(ColumnCastingError) as excinfo:
        _, _ = cast_typed_columns(df, schema)
    assert excinfo.value.error_msg == "cannot safely cast non-equivalent float64 to int64"
    assert excinfo.value.attempted_casting_columns == ["A"]
    assert excinfo.value.attempted_casting_type == "Int64"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_cast_to_float_no_error():
    test_case = CastingTestCase(
        name="cast_to_float_no_error",
        df=get_df(),
        features=TypedColumns(
            to_float=["A", "B", "C"],
            inferred=["D"],
        ),
        tags=["C", "D"],
        expected={
            "column_types": {
                # for each column key, value represents the new column type
                "A": "Float64",
                "B": "Float64",
                "C": "Float64",
                "D": "object",
            },
            "value_types": {
                # for each column key, tuple represents the new value types at index 0 and index 1
                "A": (np.float64, pd.NA),
                "B": (np.float64, pd.NA),
                "C": (np.float64, pd.NA),
                "D": (str, float),
            },
        },
    )
    table_test(test_case)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_cast_to_int_no_error():
    test_case = CastingTestCase(
        name="cast_to_int_no_error",
        df=get_df(),
        features=TypedColumns(
            to_int=["A", "B", "C"],
            inferred=["D"],
        ),
        tags=None,
        expected={
            "column_types": {
                # for each column key, value represents the new column type
                "A": "Int64",
                "B": "Int64",
                "C": "Int64",
                "D": "object",
            },
            "value_types": {
                # for each column key, tuple represents the new value types at index 0 and index 1
                "A": (np.int64, pd.NA),
                "B": (np.int64, pd.NA),
                "C": (np.int64, pd.NA),
                "D": (str, float),
            },
        },
    )
    table_test(test_case)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_cast_to_categorical_no_error():
    test_case = CastingTestCase(
        name="cast_to_categorical_no_error",
        df=get_df(),
        features=TypedColumns(
            to_str=["A", "B", "C"],
            inferred=["D"],
        ),
        tags=TypedColumns(
            inferred=["A"],
        ),
        expected={
            "column_types": {
                # for each column key, value represents the new column type
                "A": "string",
                "B": "string",
                "C": "string",
                "D": "object",
            },
            "value_types": {
                # for each column key, tuple represents the new value types at index 0 and index 1
                "A": (str, pd.NA),
                "B": (str, pd.NA),
                "C": (str, pd.NA),
                "D": (str, float),
            },
        },
    )
    table_test(test_case)


def table_test(case: CastingTestCase):
    schema = get_schema(case.features, case.tags)
    try:
        dataframe, schema = cast_typed_columns(case.df, schema)
    except Exception:
        pytest.fail("Unexpected error!")

    # column types
    for k, v in case.expected["column_types"].items():
        assert dataframe[k].dtype == v

    # value types
    for k, v in case.expected["value_types"].items():
        assert isinstance(dataframe[k][0], v[0])
        if k in case.get_all_inferred_columns():
            assert isinstance(dataframe[k][1], v[1])
        else:
            # these are all set up to be pd.NA post-casting.
            # pd.NA is not a type, so need to check the value differently.
            assert dataframe[k][1] is v[1]

    # null value handling
    nulls = dataframe.isna()
    for k in case.expected["column_types"].keys():
        assert not nulls[k][0]
        assert nulls[k][1]

    # function output
    assert isinstance(schema.feature_column_names, list)
    assert set(schema.feature_column_names) == set(case.features.get_all_column_names())


def get_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": pd.Series([int(1), np.nan]),
            "B": pd.Series([float(1.0), float("nan")]),
            "C": pd.Series([str("1"), np.nan]),
            "D": pd.Series([str("2"), np.nan]),
        },
    )


def get_schema(features: TypedColumns, tags: TypedColumns) -> Schema:
    return Schema(
        prediction_id_column_name="prediction_id",
        timestamp_column_name="prediction_timestamp",
        prediction_label_column_name="prediction_label",
        actual_label_column_name="actual_label",
        feature_column_names=features,
        tag_column_names=tags,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
