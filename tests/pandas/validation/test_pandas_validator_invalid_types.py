import datetime
from collections import ChainMap

import arize.pandas.validation.errors as err
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from arize.pandas.logger import Schema
from arize.pandas.validation.validator import Validator
from arize.utils.types import EmbeddingColumnNames, ModelTypes, ObjectDetectionColumnNames


def test_zero_errors():
    kwargs = get_kwargs()
    errors = Validator.validate_types(**kwargs)
    assert len(errors) == 0


# may need to revisit this case
def test_reject_all_nones():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(pd.DataFrame({"A": pd.Series([None])})),
            },
            kwargs,
        )
    )
    assert len(errors) == 2
    assert type(errors[0]) is err.InvalidTypeFeatures
    assert type(errors[1]) is err.InvalidTypeTags


def test_invalid_type_prediction_id():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"prediction_id": pd.Series([datetime.datetime.now()])})
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidType


def test_invalid_type_prediction_id_float():
    kwargs = get_kwargs()
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
    kwargs = get_kwargs()
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
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"prediction_timestamp": pd.Series([datetime.datetime.now()])})
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_timestamp_date():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {"prediction_timestamp": pd.Series([datetime.datetime.now().date()])}
                    )
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_timestamp_float():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {"prediction_timestamp": pd.Series([datetime.datetime.now().timestamp()])}
                    )
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_timestamp_int():
    kwargs = get_kwargs()
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
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {"pyarrow_schema": pa.Schema.from_pandas(pd.DataFrame({"A": pd.Series([list()])}))},
            kwargs,
        )
    )
    assert len(errors) == 2
    assert type(errors[0]) is err.InvalidTypeFeatures
    assert type(errors[1]) is err.InvalidTypeTags


def test_invalid_type_shap_values():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {"pyarrow_schema": pa.Schema.from_pandas(pd.DataFrame({"a": pd.Series([list()])}))},
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidTypeShapValues


def test_invalid_label():
    kwargs = get_kwargs()
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
    kwargs = get_kwargs()
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
    kwargs = get_kwargs()
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
    kwargs = get_kwargs()
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
    kwargs = get_kwargs()
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
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(pd.DataFrame({"A": pd.Series([int(1)])})),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_tag_int():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(pd.DataFrame({"A": pd.Series([int(1)])})),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_feature_str():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(pd.DataFrame({"A": pd.Series(["0"])})),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_tag_str():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(pd.DataFrame({"A": pd.Series(["0"])})),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_feature_float():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(pd.DataFrame({"A": pd.Series([3.14])})),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_tag_float():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(pd.DataFrame({"A": pd.Series([3.14])})),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_feature_bool():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(pd.DataFrame({"A": pd.Series([True])})),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_ag_bool():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(pd.DataFrame({"A": pd.Series([True])})),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_score():
    kwargs = get_kwargs()
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


def test_multiple():
    kwargs = get_kwargs()
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


def test_invalid_type_bounding_boxes():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "model_type": ModelTypes.OBJECT_DETECTION,
            },
            kwargs,
        )
    )
    expected_errors = [
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "pred_wrong_bounding_boxes_coordinates",
                "actual_wrong_bounding_boxes_coordinates",
            ],
            expected_types=["List[List[float]]"],
        ),
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "pred_wrong_bounding_boxes_categories",
                "actual_wrong_bounding_boxes_categories",
            ],
            expected_types=["List[str]"],
        ),
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "pred_wrong_bounding_boxes_scores",
                "actual_wrong_bounding_boxes_scores",
            ],
            expected_types=["List[float]"],
        ),
    ]

    assert len(errors) == len(expected_errors)
    for error, expected_error in zip(errors, expected_errors):
        assert error.error_message() == expected_error.error_message()
        assert type(error) == type(expected_error)


def test_invalid_type_prompt_response():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "model_type": ModelTypes.GENERATIVE_LLM,
            },
            kwargs,
        )
    )
    expected_errors = [
        err.InvalidTypePromptResponse(
            wrong_type_columns=[
                "wrong_prompt_vector",
                "wrong_response_vector",
            ],
            expected_types=["list[float], np.array[float]"],
        ),
        err.InvalidTypePromptResponse(
            wrong_type_columns=[
                "wrong_prompt_data",
                "wrong_response_data",
            ],
            expected_types=["list[string]"],
        ),
    ]

    assert len(errors) == len(expected_errors)
    for error, expected_error in zip(errors, expected_errors):
        assert error.error_message() == expected_error.error_message()
        assert type(error) == type(expected_error)


def get_kwargs():
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
            object_detection_prediction_column_names=ObjectDetectionColumnNames(
                bounding_boxes_coordinates_column_name="pred_wrong_bounding_boxes_coordinates",
                categories_column_name="pred_wrong_bounding_boxes_categories",
                scores_column_name="pred_wrong_bounding_boxes_scores",
            ),
            object_detection_actual_column_names=ObjectDetectionColumnNames(
                bounding_boxes_coordinates_column_name="actual_wrong_bounding_boxes_coordinates",
                categories_column_name="actual_wrong_bounding_boxes_categories",
                scores_column_name="actual_wrong_bounding_boxes_scores",
            ),
            prompt_column_names=EmbeddingColumnNames(
                vector_column_name="wrong_prompt_vector",
                data_column_name="wrong_prompt_data",
            ),
            response_column_names=EmbeddingColumnNames(
                vector_column_name="wrong_response_vector",
                data_column_name="wrong_response_data",
            ),
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
                    ##### Wrong type bounding boxes
                    "pred_wrong_bounding_boxes_coordinates": pd.Series(
                        [
                            ["wrong_type", "wrong_type"],
                        ]
                    ),
                    "pred_wrong_bounding_boxes_categories": pd.Series(
                        [
                            [1, 2],  # Wrong type
                        ]
                    ),
                    "pred_wrong_bounding_boxes_scores": pd.Series(
                        [
                            ["wrong_type", "wrong_type"],
                        ]
                    ),
                    "actual_wrong_bounding_boxes_coordinates": pd.Series(
                        [
                            ["wrong_type", "wrong_type"],
                        ]
                    ),
                    "actual_wrong_bounding_boxes_categories": pd.Series(
                        [
                            [1, 2],  # Wrong type
                        ]
                    ),
                    "actual_wrong_bounding_boxes_scores": pd.Series(
                        [
                            ["wrong_type", "wrong_type"],
                        ]
                    ),
                    # Wrong type
                    "wrong_prompt_vector": np.random.randn(3, 3).astype(str).tolist(),
                    "wrong_response_vector": np.random.choice(
                        a=[True, False], size=(3, 3)
                    ).tolist(),
                    "wrong_prompt_data": [x for x in range(3)],
                    "wrong_response_data": [x for x in range(3)],
                }
            )
        ),
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
