import datetime
from collections import ChainMap

import pandas as pd
import pytest

import arize.pandas.validation.errors as err
from arize.pandas.logger import Schema
from arize.pandas.validation.validator import Validator
from arize.utils.types import Environments, ModelTypes


def test_valid_record_score_categorical_prod():
    kwargs = get_sc_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.PRODUCTION,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(
                            ["fraud", "not fraud", None]
                        ),
                        "prediction_score": pd.Series([0.2, 0.3, float("NaN")]),
                        "actual_label": pd.Series(["not fraud", "fraud", None]),
                        "actual_score": pd.Series([0, float("NaN"), 1]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_record_score_categorical_prod():
    kwargs = get_sc_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.PRODUCTION,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(
                            ["fraud", "not fraud", None]
                        ),
                        "prediction_score": pd.Series([0.2, 0.3, float("NaN")]),
                        "actual_label": pd.Series(["not fraud", "fraud", None]),
                        "actual_score": pd.Series([0, 1, float("NaN")]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == [
        "prediction_label",
        "prediction_score",
        "actual_label",
        "actual_score",
    ]
    assert errors[0].indexes == [2]


def test_invalid_record_score_categorical_pred_only_prod():
    kwargs = get_sc_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.PRODUCTION,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(
                            ["fraud", "not fraud", None]
                        ),
                        "prediction_score": pd.Series([0.2, 0.3, float("NaN")]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["prediction_label", "prediction_score"]
    assert errors[0].indexes == [2]


def test_invalid_record_score_categorical_actual_only_prod():
    kwargs = get_sc_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.PRODUCTION,
                "dataframe": pd.DataFrame(
                    {
                        "actual_label": pd.Series(["not fraud", "fraud", None]),
                        "actual_score": pd.Series([0, 1, float("NaN")]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["actual_label", "actual_score"]
    assert errors[0].indexes == [2]


def test_valid_record_numeric_label_prod():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.PRODUCTION,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series([0.2, 0.3, float("NaN")]),
                        "actual_label": pd.Series([0.1, float("NaN"), 0.2]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_record_numeric_score_prod():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.PRODUCTION,
                "schema": Schema(
                    prediction_score_column_name="prediction_score",
                    actual_score_column_name="actual_score",
                ),
                "dataframe": pd.DataFrame(
                    {
                        "prediction_score": pd.Series([0.2, 0.3, float("NaN")]),
                        "actual_score": pd.Series([0.1, float("NaN"), 0.2]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_record_numeric_mix_prod():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.PRODUCTION,
                "schema": Schema(
                    prediction_label_column_name="prediction_label",
                    actual_score_column_name="actual_score",
                ),
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series([0.2, 0.3, float("NaN")]),
                        "actual_score": pd.Series([0.1, float("NaN"), 0.2]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_record_numeric_pred_score_prod():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.PRODUCTION,
                "dataframe": pd.DataFrame(
                    {
                        "pred_score": pd.Series([0.1, 0.3, 0.2]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_record_numeric_prod():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.PRODUCTION,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series([0.2, 0.3, float("NaN")]),
                        "actual_label": pd.Series([0.1, 0.2, float("NaN")]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["prediction_label", "actual_label"]
    assert errors[0].indexes == [2]


def test_invalid_record_numeric_pred_only_prod():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.PRODUCTION,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series([0.2, 0.3, float("NaN")]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["prediction_label"]
    assert errors[0].indexes == [2]


def test_invalid_record_numeric_actual_only_prod():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.PRODUCTION,
                "dataframe": pd.DataFrame(
                    {
                        "actual_label": pd.Series([0.1, 0.2, float("NaN")]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["actual_label"]
    assert errors[0].indexes == [2]


def test_valid_record_score_categorical_training():
    kwargs = get_sc_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(
                            ["fraud", "not fraud", None]
                        ),
                        "prediction_score": pd.Series([0.2, float("NaN"), 0.3]),
                        "actual_label": pd.Series(["not fraud", "fraud", None]),
                        "actual_score": pd.Series([0, float("NaN"), 1]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_record_score_categorical_training():
    kwargs = get_sc_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(
                            ["fraud", "not fraud", None]
                        ),
                        "prediction_score": pd.Series([0.2, 0.3, float("NaN")]),
                        "actual_label": pd.Series(["not fraud", "fraud", None]),
                        "actual_score": pd.Series([0, 1, float("NaN")]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 2
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["prediction_label", "prediction_score"]
    assert errors[0].indexes == [2]
    assert type(errors[1]) is err.InvalidRecord
    assert errors[1].columns == ["actual_label", "actual_score"]
    assert errors[1].indexes == [2]


def test_invalid_record_score_categorical_pred_only_training():
    kwargs = get_sc_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series(
                            ["fraud", "not fraud", None]
                        ),
                        "prediction_score": pd.Series([0.2, 0.3, float("NaN")]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["prediction_label", "prediction_score"]
    assert errors[0].indexes == [2]


def test_invalid_record_score_categorical_actual_only_training():
    kwargs = get_sc_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "dataframe": pd.DataFrame(
                    {
                        "actual_label": pd.Series(["not fraud", "fraud", None]),
                        "actual_score": pd.Series([0, 1, float("NaN")]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["actual_label", "actual_score"]
    assert errors[0].indexes == [2]


def test_valid_record_numeric_label_training():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series([0.2, 0.3, 0.5]),
                        "actual_label": pd.Series([0.1, 0.5, 0.2]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_record_numeric_label_training():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "dataframe": pd.DataFrame(
                    {
                        "prediction_label": pd.Series([0.2, 0.3, float("NaN")]),
                        "actual_label": pd.Series([0.1, float("NaN"), 0.2]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 2
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["prediction_label"]
    assert errors[0].indexes == [2]
    assert type(errors[1]) is err.InvalidRecord
    assert errors[1].columns == ["actual_label"]
    assert errors[1].indexes == [1]


def test_valid_record_numeric_score_training():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "schema": Schema(
                    prediction_score_column_name="prediction_score",
                    actual_score_column_name="actual_score",
                ),
                "dataframe": pd.DataFrame(
                    {
                        "prediction_score": pd.Series([0.2, 0.3, 0.5]),
                        "actual_score": pd.Series([0.1, 0.5, 0.2]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_record_numeric_score_training():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "schema": Schema(
                    prediction_score_column_name="prediction_score",
                    actual_score_column_name="actual_score",
                ),
                "dataframe": pd.DataFrame(
                    {
                        "prediction_score": pd.Series([0.2, 0.3, float("NaN")]),
                        "actual_score": pd.Series([0.1, float("NaN"), 0.2]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 2
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["prediction_score"]
    assert errors[0].indexes == [2]
    assert type(errors[1]) is err.InvalidRecord
    assert errors[1].columns == ["actual_score"]
    assert errors[1].indexes == [1]


def test_valid_record_numeric_score_label_training():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "schema": Schema(
                    prediction_score_column_name="prediction_score",
                    actual_label_column_name="actual_label",
                ),
                "dataframe": pd.DataFrame(
                    {
                        "prediction_score": pd.Series([0.2, 0.3, 0.5]),
                        "actual_label": pd.Series([0, 1, 1]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_record_numeric_score_label_training():
    kwargs = get_numeric_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "schema": Schema(
                    prediction_score_column_name="prediction_score",
                    actual_label_column_name="actual_label",
                ),
                "dataframe": pd.DataFrame(
                    {
                        "prediction_score": pd.Series([0.2, 0.3, float("NaN")]),
                        "actual_label": pd.Series([0, 1, 1]),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidRecord
    assert errors[0].columns == ["prediction_score"]
    assert errors[0].indexes == [2]


def get_sc_kwargs():
    return {
        "model_type": ModelTypes.SCORE_CATEGORICAL,
        "schema": Schema(
            prediction_id_column_name="prediction_id",
            timestamp_column_name="prediction_timestamp",
            prediction_label_column_name="prediction_label",
            actual_label_column_name="actual_label",
            prediction_score_column_name="prediction_score",
            actual_score_column_name="actual_score",
            feature_column_names=list("ABCD"),
            tag_column_names=list("ABCD"),
            shap_values_column_names=dict(zip("ABCD", "abcd")),
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
                #####
                "A": pd.Series([0, 1, 2]),
                "B": pd.Series([0.0, 1.0, 2.0]),
                "C": pd.Series([float("NaN"), float("NaN"), float("NaN")]),
                #####
                "a": pd.Series([0, 1, 2]),
                "b": pd.Series([0.0, 1.0, 2.0]),
                "c": pd.Series([float("NaN"), float("NaN"), float("NaN")]),
            }
        ),
    }


def get_numeric_kwargs():
    return {
        "model_type": ModelTypes.NUMERIC,
        "schema": Schema(
            prediction_id_column_name="prediction_id",
            timestamp_column_name="prediction_timestamp",
            prediction_label_column_name="prediction_label",
            actual_label_column_name="actual_label",
            feature_column_names=list("ABCD"),
            tag_column_names=list("ABCD"),
            shap_values_column_names=dict(zip("ABCD", "abcd")),
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
                #####
                "A": pd.Series([0, 1, 2]),
                "B": pd.Series([0.0, 1.0, 2.0]),
                "C": pd.Series([float("NaN"), float("NaN"), float("NaN")]),
                #####
                "a": pd.Series([0, 1, 2]),
                "b": pd.Series([0.0, 1.0, 2.0]),
                "c": pd.Series([float("NaN"), float("NaN"), float("NaN")]),
            }
        ),
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
