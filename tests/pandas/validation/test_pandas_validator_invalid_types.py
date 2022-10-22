import datetime
from collections import ChainMap

import pandas as pd
import pyarrow as pa
import pytest

import arize.pandas.validation.errors as err
from arize.pandas.logger import Schema
from arize.pandas.validation.validator import Validator
from arize.utils.types import ModelTypes


def test_zero_errors():
    errors = Validator.validate_types(**kwargs)
    assert len(errors) == 0


# may need to revisit this case
def test_reject_all_nones():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([None])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 2
    assert type(errors[0]) is err.InvalidTypeFeatures
    assert type(errors[1]) is err.InvalidTypeTags


def test_invalid_type_prediction_id():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {"prediction_id": pd.Series([datetime.datetime.now()])}
                    )
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidType


def test_invalid_type_prediction_id_float():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"prediction_id": pd.Series([3.14])})
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidType


def test_invalid_type_timestamp():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"prediction_timestamp": pd.Series(["now"])})
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidType


def test_valid_timestamp_datetime():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {"prediction_timestamp": pd.Series([datetime.datetime.now()])}
                    )
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_timestamp_date():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {
                            "prediction_timestamp": pd.Series(
                                [datetime.datetime.now().date()]
                            )
                        }
                    )
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_timestamp_float():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {
                            "prediction_timestamp": pd.Series(
                                [datetime.datetime.now().timestamp()]
                            )
                        }
                    )
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_timestamp_int():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {
                            "prediction_timestamp": pd.Series(
                                [int(datetime.datetime.now().timestamp())]
                            )
                        }
                    )
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_type_dimensions():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([list()])})
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 2
    assert type(errors[0]) is err.InvalidTypeFeatures
    assert type(errors[1]) is err.InvalidTypeTags


def test_invalid_type_shap_values():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"a": pd.Series([list()])})
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidTypeShapValues


def test_invalid_label():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"prediction_label": pd.Categorical([None])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidType


def test_valid_label_int():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"prediction_label": pd.Series([int(1)])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_label_str():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"prediction_label": pd.Series(["0"])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_label_float():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"prediction_label": pd.Series([3.14])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_label_bool():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"prediction_label": pd.Series([True])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_feature_int():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([int(1)])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_tag_int():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([int(1)])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_feature_str():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series(["0"])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_tag_str():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series(["0"])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_feature_float():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([3.14])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_tag_float():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([3.14])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_feature_bool():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([True])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_ag_bool():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([True])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_score():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"prediction_score": pd.Series(["fraud"])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidType


def test_invalid_num_seq_not_list():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"act_num_seq": pd.Series([0.1])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidType


# allow all None
def test_valid_num_seq_list_all_none():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"act_num_seq": pd.Series([None])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


# allow all NaN when dtype is object
def test_valid_num_seq_list_all_nan():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"act_num_seq": [float("NaN")]}, dtype=object)
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


# allow all empty
def test_valid_num_seq_list_all_empty():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"act_num_seq": pd.Series([[]])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


# allow all None as elements
def test_valid_num_seq_list_all_none_elements():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"act_num_seq": pd.Series([[None]])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_num_seq_list_not_numeric():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"act_num_seq": pd.Series([[[]]])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidType


def test_multiple():
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {
                            "prediction_id": pd.Series([datetime.datetime.now()]),
                            "prediction_timestamp": pd.Series(["now"]),
                            "A": pd.Series([list()]),
                            "a": pd.Series([list()]),
                            "prediction_label": pd.Categorical([None]),
                            "prediction_score": pd.Series(["fraud"]),
                        }
                    )
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 7
    assert any(type(e) is err.InvalidType for e in errors)
    assert any(type(e) is err.InvalidTypeFeatures for e in errors)
    assert any(type(e) is err.InvalidTypeTags for e in errors)
    assert any(type(e) is err.InvalidTypeShapValues for e in errors)


kwargs = {
    "model_type": ModelTypes.SCORE_CATEGORICAL,
    "schema": Schema(
        prediction_id_column_name="prediction_id",
        timestamp_column_name="prediction_timestamp",
        prediction_label_column_name="prediction_label",
        actual_label_column_name="actual_label",
        prediction_score_column_name="prediction_score",
        actual_score_column_name="actual_score",
        feature_column_names=list("ABCDEFG"),
        tag_column_names=list("ABCDEFG"),
        shap_values_column_names=dict(zip("ABCDEF", "abcdef")),
        actual_numeric_sequence_column_name="act_num_seq",
    ),
    "pyarrow_schema": pa.Schema.from_pandas(
        pd.DataFrame(
            {
                "prediction_id": pd.Series(["0", "1", "2"]),
                "prediction_timestamp": pd.Series(
                    [
                        datetime.datetime.now(),
                        datetime.datetime.now(),
                        datetime.datetime.now(),
                    ]
                ),
                "prediction_label": pd.Series(["fraud", "not fraud", "fraud"]),
                "prediction_score": pd.Series([0.2, 0.3, 0.4]),
                "actual_label": pd.Series(["not fraud", "fraud", "not fraud"]),
                "actual_score": pd.Series([0, 1, 0]),
                "act_num_seq": pd.Series([[], None, [None, 0]]),
                #####
                "A": pd.Series([0, 1, 2]),
                "B": pd.Series([0.0, 1.0, 2.0]),
                "C": pd.Series([float("NaN"), float("NaN"), float("NaN")]),
                "D": pd.Series([0, float("NaN"), 2]),
                "E": pd.Series([0, None, 2]),
                "F": pd.Series([None, float("NaN"), None]),
                "G": pd.Series(["foo", "bar", "baz"]),
                "H": pd.Series([True, False, True]),
                #####
                "a": pd.Series([0, 1, 2]),
                "b": pd.Series([0.0, 1.0, 2.0]),
                "c": pd.Series([float("NaN"), float("NaN"), float("NaN")]),
                "d": pd.Series([0, float("NaN"), 2]),
                "e": pd.Series([0, None, 2]),
                "f": pd.Series([None, float("NaN"), None]),
            }
        )
    ),
}

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
