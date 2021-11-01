import pytest
import pandas as pd
from collections import ChainMap
import datetime

from arize.pandas.logger import Schema
from arize.pandas.validation.validator import Validator
import arize.pandas.validation.errors as err


def test_zero_errors():
    errors = Validator.validate_values(**kwargs)
    assert len(errors) == 0


def test_invalid_ts_missing_value():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [
                                (
                                    datetime.datetime.now()
                                    - datetime.timedelta(days=365)
                                ).date(),
                                float("NaN"),
                            ]
                        )
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_valid_ts_empty_df():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {"prediction_timestamp": pd.Series([], dtype=float)},
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_ts_date32_min():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [
                                (
                                    datetime.datetime.now()
                                    - datetime.timedelta(days=365)
                                ).date()
                            ]
                        )
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueTimestamp


def test_invalid_ts_date32_max():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [
                                (
                                    datetime.datetime.now()
                                    + datetime.timedelta(
                                        days=366
                                    )  # need to fudge a little b/c time is always moving forward
                                ).date()
                            ]
                        )
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueTimestamp


def test_invalid_ts_float64_min():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [
                                (
                                    datetime.datetime.now()
                                    - datetime.timedelta(days=365)
                                ).timestamp()
                            ]
                        )
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueTimestamp


def test_invalid_ts_float64_max():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [
                                (
                                    datetime.datetime.now()
                                    + datetime.timedelta(
                                        days=366
                                    )  # need to fudge a little b/c time is always moving forward
                                ).timestamp()
                            ]
                        )
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueTimestamp


def test_invalid_ts_int64_min():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [
                                int(
                                    (
                                        datetime.datetime.now()
                                        - datetime.timedelta(days=365)
                                    ).timestamp()
                                )
                            ]
                        )
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueTimestamp


def test_invalid_ts_int64_max():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [
                                int(
                                    (
                                        datetime.datetime.now()
                                        + datetime.timedelta(
                                            days=366
                                        )  # need to fudge a little b/c time is always moving forward
                                    ).timestamp()
                                )
                            ]
                        )
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueTimestamp


def test_invalid_ts_datetime_min():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [(datetime.datetime.now() - datetime.timedelta(days=365))]
                        )
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueTimestamp


def test_invalid_ts_datetime_max():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [
                                datetime.datetime.now()
                                + datetime.timedelta(
                                    days=366
                                )  # need to fudge a little b/c time is always moving forward
                            ]
                        )
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueTimestamp


def test_invalid_prediction_label_none_value():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {"prediction_label": pd.Series(["foo", None, "baz"])}
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_invalid_actual_label_none_value():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {"actual_label": pd.Series(["foo", None, "baz"])}
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_invalid_prediction_label_nan_value():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {"prediction_label": pd.Series([0, float("NaN"), 1])}
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_invalid_actual_label_nan_value():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {"actual_label": pd.Series([0, float("NaN"), 1])}
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_invalid_prediction_label_inf_value():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {"prediction_label": pd.Series([0, float("-inf"), 1])}
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_invalid_actual_label_inf_value():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {"actual_label": pd.Series([0, float("-inf"), 1])}
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_invalid_prediction_id_none():
    errors = Validator.validate_values(
        **ChainMap(
            {"dataframe": pd.DataFrame({"prediction_id": pd.Series(["0", None, "1"])})},
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_multiple():
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [
                                (
                                    datetime.datetime.now()
                                    - datetime.timedelta(days=365)
                                ).date()
                            ]
                            * 3
                        ),
                        "prediction_label": pd.Series(["foo", None, "baz"]),
                        "actual_label": pd.Series([0, float("NaN"), 1]),
                    }
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 3
    assert any(type(e) is err.InvalidValueTimestamp for e in errors)
    assert any(type(e) is err.InvalidValueMissingValue for e in errors)


kwargs = {
    "schema": Schema(
        prediction_id_column_name="prediction_id",
        timestamp_column_name="prediction_timestamp",
        prediction_label_column_name="prediction_label",
        actual_label_column_name="actual_label",
        prediction_score_column_name="prediction_score",
        actual_score_column_name="actual_score",
        feature_column_names=list("ABCDEFG"),
        shap_values_column_names=dict(zip("ABCDEF", "abcdef")),
    ),
    "dataframe": pd.DataFrame(
        {
            "prediction_id": pd.Series(["0", "1", "2"]),
            "prediction_timestamp": pd.Series(
                [
                    datetime.datetime.now(),
                    datetime.datetime.now() - datetime.timedelta(days=364),
                    datetime.datetime.now() + datetime.timedelta(days=364),
                ]
            ),
            "prediction_label": pd.Series(["fraud", "not fraud", "fraud"]),
            "prediction_score": pd.Series([0.2, 0.3, 0.4]),
            "actual_label": pd.Series(["not fraud", "fraud", "not fraud"]),
            "actual_score": pd.Series([0, 1, 0]),
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
    ),
}

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
