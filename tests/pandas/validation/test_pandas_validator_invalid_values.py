import datetime
from collections import ChainMap

import numpy as np
import pandas as pd
import pytest
from arize.pandas.validation import errors as err
from arize.pandas.validation.validator import Validator
from arize.utils.types import EmbeddingColumnNames, ModelTypes, ObjectDetectionColumnNames, Schema


def test_zero_errors():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(**kwargs)
    assert len(errors) == 0


def test_invalid_ts_missing_value():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [
                                (datetime.datetime.now() - datetime.timedelta(days=365)).date(),
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
    kwargs = get_standard_kwargs()
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
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [(datetime.datetime.now() - datetime.timedelta(days=365)).date()]
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
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [
                                (
                                    datetime.datetime.now()
                                    + datetime.timedelta(days=366)
                                    # need to fudge a little b/c time is always moving forward
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
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [(datetime.datetime.now() - datetime.timedelta(days=365)).timestamp()]
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
    kwargs = get_standard_kwargs()
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
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [
                                int(
                                    (
                                        datetime.datetime.now() - datetime.timedelta(days=365)
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
    kwargs = get_standard_kwargs()
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
                                        )  # need to fudge a little b/c time always moves forward
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
    kwargs = get_standard_kwargs()
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
    kwargs = get_standard_kwargs()
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
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {"dataframe": pd.DataFrame({"prediction_label": pd.Series(["foo", None, "baz"])})},
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_invalid_actual_label_none_value():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {"dataframe": pd.DataFrame({"actual_label": pd.Series(["foo", None, "baz"])})},
            kwargs,
        )
    )
    msgs = [e.error_message() for e in errors]
    assert len(errors) == 1, f"msg is {msgs}"
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_invalid_prediction_label_nan_value():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {"dataframe": pd.DataFrame({"prediction_label": pd.Series([0, float("NaN"), 1])})},
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_invalid_actual_label_nan_value():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {"dataframe": pd.DataFrame({"actual_label": pd.Series([0, float("NaN"), 1])})},
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_invalid_prediction_label_inf_value():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {"dataframe": pd.DataFrame({"prediction_label": pd.Series([0, float("-inf"), 1])})},
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_invalid_actual_label_inf_value():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {"dataframe": pd.DataFrame({"actual_label": pd.Series([0, float("-inf"), 1])})},
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_invalid_prediction_id_none():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {"dataframe": pd.DataFrame({"prediction_id": pd.Series(["0", None, "1"])})},
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueMissingValue


def test_prediction_id_length():
    long_ids = pd.Series(["A" * 129] * 4)
    empty_ids = pd.Series([""] * 4)
    kwargs = get_standard_kwargs()
    good_vector = []
    for i in range(4):
        good_vector.append(np.arange(float(6)))

    for ids in (long_ids, empty_ids):
        errors = Validator.validate_values(
            **ChainMap(
                {
                    "schema": Schema(
                        prediction_id_column_name="prediction_id",
                        embedding_feature_column_names={
                            "good_vector": EmbeddingColumnNames(
                                vector_column_name="good_vector",  # Should not give error
                            ),
                        },
                    ),
                },
                {"dataframe": pd.DataFrame({"prediction_id": ids, "good_vector": good_vector})},
                kwargs,
            )
        )
        assert len(errors) == 1
        assert type(errors[0]) is err.InvalidStringLength
        err_string = (
            "prediction_id_column_name column prediction_id contains invalid values. "
            "Only string values of length within 1 - 128 are accepted."
        )
        assert errors[0].error_message() == err_string


def test_multiple():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_values(
        **ChainMap(
            {
                "dataframe": pd.DataFrame(
                    {
                        "prediction_timestamp": pd.Series(
                            [(datetime.datetime.now() - datetime.timedelta(days=365)).date()] * 3
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


def test_invalid_embedding_dimensionality():
    kwargs = get_standard_kwargs()
    good_vector = []
    for i in range(4):
        good_vector.append(np.arange(float(6)))

    multidimensional_vector = []
    for i in range(4):
        if i <= 1:
            multidimensional_vector.append(np.arange(float(6)))
        else:
            multidimensional_vector.append(np.arange(float(4)))

    one_vector = []
    for i in range(4):
        one_vector.append(np.arange(float(1)))

    null_vector = []
    null_vector.append(np.arange(float(3)))
    null_vector.append(None)
    null_vector.append(np.NaN)
    null_vector.append([])

    errors = Validator.validate_values(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    embedding_feature_column_names={
                        "good_vector": EmbeddingColumnNames(
                            vector_column_name="good_vector",  # Should not give error
                        ),
                        "multidimensional_vector": EmbeddingColumnNames(
                            vector_column_name="multidimensional_vector",  # Should not give error
                        ),
                        "one_vector": EmbeddingColumnNames(
                            vector_column_name="one_vector",  # Should give error
                        ),
                        "null_vector": EmbeddingColumnNames(
                            vector_column_name="null_vector",  # Should not give error
                        ),
                    },
                ),
                "dataframe": pd.DataFrame(
                    {
                        "good_vector": good_vector,
                        "multidimensional_vector": multidimensional_vector,
                        "one_vector": one_vector,
                        "null_vector": null_vector,
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidValueLowEmbeddingVectorDimensionality


def test_invalid_value_bounding_boxes_coordinates():
    kwargs = get_object_detection_kwargs()
    # Success case
    errors = Validator.validate_values(**kwargs)
    assert len(errors) == 0

    # Box' coordinates list with None value
    dataframe = kwargs["dataframe"]
    dataframe["prediction_bounding_boxes_coordinates"] = pd.Series(
        [
            None,
            [],
            [[0.31, 0.32, 0.33, 0.34], [0.31, 0.32, 0.33, 0.34]],
        ]
    )
    for i in range(len(dataframe)):  # Go row by row checking errors
        df = dataframe.iloc[[i]]
        errors = Validator.validate_values(
            **ChainMap(
                {
                    "dataframe": df,
                },
                kwargs,
            )
        )
        if i == 0:
            assert len(errors) == 1
            expected_error = err.InvalidBoundingBoxesCoordinates(reason="none_boxes")
            error = errors[0]
            assert type(error) == type(expected_error)
            assert error.error_message() == expected_error.error_message()
        else:
            assert len(errors) == 0

    # Empty boxes case
    dataframe = kwargs["dataframe"]
    dataframe["prediction_bounding_boxes_coordinates"] = pd.Series(
        [
            [[], [0.11, 0.12, 0.13, 0.14]],
            [[0.21, 0.22, 0.23, 0.24], None],
            [[0.31, 0.32, 0.33, 0.34], [0.31, 0.32, 0.33, 0.34]],
        ]
    )
    for i in range(len(dataframe)):  # Go row by row checking errors
        df = dataframe.iloc[[i]]
        errors = Validator.validate_values(
            **ChainMap(
                {
                    "dataframe": df,
                },
                kwargs,
            )
        )
        if i == 2:
            assert len(errors) == 0
        else:
            assert len(errors) == 1
            expected_error = err.InvalidBoundingBoxesCoordinates(reason="none_or_empty_box")
            error = errors[0]
            assert type(error) == type(expected_error)
            assert error.error_message() == expected_error.error_message()

    # Box with wrong format
    dataframe = kwargs["dataframe"]
    dataframe["prediction_bounding_boxes_coordinates"] = pd.Series(
        [
            [[0.11, 0.12, 0.13, 0.14, 0.15], [0.11, 0.12, 0.13, 0.14]],  # 5 coordinate values
            [[-0.21, 0.22, 0.23, 0.24], [0.21, 0.22, 0.23, 0.24]],  # coordinate value < 0
            [[0.31, 0.32, 0.33, 0.34], [1.31, 0.32, 0.33, 0.34]],  # coordinate value > 1
        ]
    )
    for i in range(len(dataframe)):  # Go row by row checking errors
        df = dataframe.iloc[[i]]
        errors = Validator.validate_values(
            **ChainMap(
                {
                    "dataframe": df,
                },
                kwargs,
            )
        )
        assert len(errors) == 1
        expected_error = err.InvalidBoundingBoxesCoordinates(
            reason="boxes_coordinates_wrong_format"
        )
        error = errors[0]
        assert type(error) == type(expected_error)
        assert error.error_message() == expected_error.error_message()


def test_invalid_value_bounding_boxes_categories():
    kwargs = get_object_detection_kwargs()
    # Success case
    errors = Validator.validate_values(**kwargs)
    assert len(errors) == 0

    # Empty category list
    dataframe = kwargs["dataframe"]
    dataframe["prediction_bounding_boxes_categories"] = pd.Series(
        [
            None,
            [],
            ["elephant", "hippo"],
        ]
    )
    for i in range(len(dataframe)):  # Go row by row checking errors
        df = dataframe.iloc[[i]]
        errors = Validator.validate_values(
            **ChainMap(
                {
                    "dataframe": df,
                },
                kwargs,
            )
        )
        if i == 0:
            assert len(errors) == 1
            expected_error = err.InvalidBoundingBoxesCategories(reason="none_category_list")
            error = errors[0]
            assert type(error) == type(expected_error)
            assert error.error_message() == expected_error.error_message()
        else:
            assert len(errors) == 0

    # Empty categories
    dataframe = kwargs["dataframe"]
    dataframe["prediction_bounding_boxes_categories"] = pd.Series(
        [
            [None, "cat"],
            ["lion", ""],
            ["elephant", "hippo"],
        ]
    )
    for i in range(len(dataframe)):  # Go row by row checking errors
        df = dataframe.iloc[[i]]
        errors = Validator.validate_values(
            **ChainMap(
                {
                    "dataframe": df,
                },
                kwargs,
            )
        )
        if i == 0:
            assert len(errors) == 1
            expected_error = err.InvalidBoundingBoxesCategories(reason="none_category")
            error = errors[0]
            assert type(error) == type(expected_error)
            assert error.error_message() == expected_error.error_message()
        else:
            assert len(errors) == 0


def test_invalid_value_bounding_boxes_scores():
    kwargs = get_object_detection_kwargs()
    # Success case
    errors = Validator.validate_values(**kwargs)
    assert len(errors) == 0

    # Empty confidence score list
    dataframe = kwargs["dataframe"]
    dataframe["prediction_bounding_boxes_scores"] = pd.Series(
        [
            None,
            [],
            [0.38, 0.73],
        ]
    )
    for i in range(len(dataframe)):  # Go row by row checking errors
        df = dataframe.iloc[[i]]
        errors = Validator.validate_values(
            **ChainMap(
                {
                    "dataframe": df,
                },
                kwargs,
            )
        )
        if i == 0:
            assert len(errors) == 1
            expected_error = err.InvalidBoundingBoxesScores(reason="none_score_list")
            error = errors[0]
            assert type(error) == type(expected_error)
            assert error.error_message() == expected_error.error_message()
        else:
            assert len(errors) == 0

    # Confidence score out of bounds
    dataframe = kwargs["dataframe"]
    dataframe["prediction_bounding_boxes_scores"] = pd.Series(
        [
            [-0.18, 0.93],
            [0.28, 1.83],
            [0.38, 0.73],
        ]
    )
    for i in range(len(dataframe)):  # Go row by row checking errors
        df = dataframe.iloc[[i]]
        errors = Validator.validate_values(
            **ChainMap(
                {
                    "dataframe": df,
                },
                kwargs,
            )
        )
        if i == 2:
            assert len(errors) == 0
        else:
            assert len(errors) == 1
            expected_error = err.InvalidBoundingBoxesScores(reason="scores_out_of_bounds")
            error = errors[0]
            assert type(error) == type(expected_error)
            assert error.error_message() == expected_error.error_message()


def get_standard_kwargs():
    return {
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


def get_object_detection_kwargs():
    return {
        "model_type": ModelTypes.OBJECT_DETECTION,
        "dataframe": pd.DataFrame(
            {
                "prediction_id": pd.Series(["0", "1", "2"]),
                "prediction_bounding_boxes_coordinates": pd.Series(
                    [
                        [[0.11, 0.12, 0.13, 0.14], [0.11, 0.12, 0.13, 0.14]],
                        [[0.21, 0.22, 0.23, 0.24], [0.21, 0.22, 0.23, 0.24]],
                        [[0.31, 0.32, 0.33, 0.34], [0.31, 0.32, 0.33, 0.34]],
                    ]
                ),
                "prediction_bounding_boxes_categories": pd.Series(
                    [
                        ["dog", "cat"],
                        ["lion", "tiger"],
                        ["elephant", "hippo"],
                    ]
                ),
                "prediction_bounding_boxes_scores": pd.Series(
                    [
                        [0.18, 0.93],
                        [0.28, 0.83],
                        [0.38, 0.73],
                    ]
                ),
                "actual_bounding_boxes_coordinates": pd.Series(
                    [
                        [[0.11, 0.12, 0.13, 0.14], [0.11, 0.12, 0.13, 0.14]],
                        [[0.21, 0.22, 0.23, 0.24], [0.21, 0.22, 0.23, 0.24]],
                        [[0.31, 0.32, 0.33, 0.34], [0.31, 0.32, 0.33, 0.34]],
                    ]
                ),
                "actual_bounding_boxes_categories": pd.Series(
                    [
                        ["dog", "cat"],
                        ["lion", "tiger"],
                        ["elephant", "hippo"],
                    ]
                ),
            }
        ),
        "schema": Schema(
            prediction_id_column_name="prediction_id",
            object_detection_prediction_column_names=ObjectDetectionColumnNames(
                bounding_boxes_coordinates_column_name="prediction_bounding_boxes_coordinates",
                categories_column_name="prediction_bounding_boxes_categories",
                scores_column_name="prediction_bounding_boxes_scores",
            ),
            object_detection_actual_column_names=ObjectDetectionColumnNames(
                bounding_boxes_coordinates_column_name="actual_bounding_boxes_coordinates",
                categories_column_name="actual_bounding_boxes_categories",
            ),
        ),
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
