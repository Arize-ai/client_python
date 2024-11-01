import datetime
from collections import ChainMap

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import arize.pandas.validation.errors as err
from arize.pandas.validation.validator import Validator
from arize.utils.types import (
    CorpusSchema,
    EmbeddingColumnNames,
    LLMConfigColumnNames,
    ModelTypes,
    ObjectDetectionColumnNames,
    PromptTemplateColumnNames,
    Schema,
)


def test_zero_errors():
    kwargs = get_kwargs()
    errors = Validator.validate_types(**kwargs)
    assert len(errors) == 0


# may need to revisit this case
def test_accept_all_nones_features_tags():
    kwargs = get_kwargs()
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
    assert len(errors) == 0


def test_accept_all_nones_labels():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {
                            "prediction_label": pd.Series([None, None, None]),
                        }
                    )
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_accept_all_nones_scores():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {
                            "actual_score": pd.Series([None, None, None]),
                        }
                    )
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_type_prediction_id():
    model_type = ModelTypes.SCORE_CATEGORICAL
    schema = Schema(
        prediction_id_column_name="prediction_id",
        timestamp_column_name="prediction_timestamp",
        prediction_label_column_name="prediction_label",
        actual_label_column_name="actual_label",
        prediction_score_column_name="prediction_score",
        actual_score_column_name="actual_score",
    )
    ts = datetime.datetime.now()
    df = pd.DataFrame(
        {
            "prediction_id": pd.Series(["0", "1", "2"]),
            "prediction_timestamp": pd.Series([ts, ts, ts]),
            "prediction_label": pd.Series(["fraud", "not fraud", "fraud"]),
            "prediction_score": pd.Series([0.2, 0.3, 0.4]),
            "actual_label": pd.Series(["not fraud", "fraud", "not fraud"]),
            "actual_score": pd.Series([0, 1, 0]),
        }
    )
    # Test case - prediction id of correct type
    pyarrow_schema = pa.Schema.from_pandas(df)
    errors = Validator.validate_types(model_type, schema, pyarrow_schema)
    assert len(errors) == 0
    # Test case - prediction id of incorrect type
    df["prediction_id"] = df["prediction_timestamp"]
    pyarrow_schema = pa.Schema.from_pandas(df)
    errors = Validator.validate_types(model_type, schema, pyarrow_schema)
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidType
    # Test case - prediction id not provided
    schema = schema.replace(prediction_id_column_name=None)
    df.drop(columns=["prediction_id"], inplace=True)
    pyarrow_schema = pa.Schema.from_pandas(df)
    errors = Validator.validate_types(model_type, schema, pyarrow_schema)
    assert len(errors) == 0


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
                    pd.DataFrame(
                        {
                            "prediction_timestamp": pd.Series(
                                [datetime.datetime.now()]
                            )
                        }
                    )
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
    kwargs = get_kwargs()
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
    kwargs = get_kwargs()
    invalid_shap_values = (
        pd.Series([list()]),
        pd.Series([None]),
    )
    for value in invalid_shap_values:
        errors = Validator.validate_types(
            **ChainMap(
                {
                    "pyarrow_schema": pa.Schema.from_pandas(
                        pd.DataFrame({"a": value})
                    )
                },
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


def test_valid_null_label():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"prediction_label": pd.Series([None])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_label_int():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"prediction_label": pd.Series([1])})
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
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([1])})
                ),
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
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([1])})
                ),
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
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series(["0"])})
                ),
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
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series(["0"])})
                ),
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
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([3.14])})
                ),
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
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([3.14])})
                ),
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
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame({"A": pd.Series([True])})
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_valid_feature_list_of_string():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {
                            "H": pd.Series(
                                [
                                    ["apple", "banana"],
                                    ["apple", "banana"],
                                    ["apple", "banana"],
                                    ["apple", "banana"],
                                    ["apple", "banana"],
                                ]
                            )
                        }
                    )
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 0


def test_invalid_feature_list_of_int():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {
                            "H": pd.Series(
                                [[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]]
                            )
                        }
                    )
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1


def test_invalid_tag_list_of_string():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {
                            "A": pd.Series(
                                [
                                    ["apple", "banana"],
                                    ["apple", "banana"],
                                    ["apple", "banana"],
                                    ["apple", "banana"],
                                    ["apple", "banana"],
                                ]
                            )
                        }
                    )
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1


def test_valid_ag_bool():
    kwargs = get_kwargs()
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
                            "prediction_id": pd.Series(
                                [datetime.datetime.now()]
                            ),
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
        )  # type:ignore
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
        assert type(error) is type(expected_error)


def test_invalid_type_generative():
    kwargs = get_kwargs()
    # prompt type: EmbeddingColumnNames
    # response type: EmbeddingColumnNames
    errors = Validator.validate_types(
        **ChainMap(
            {
                "model_type": ModelTypes.GENERATIVE_LLM,
            },
            kwargs,
        )  # type:ignore
    )
    expected_errors = [
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "wrong_prompt_vector",
                "wrong_response_vector",
            ],
            expected_types=["list[float], np.array[float]"],
        ),
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "wrong_prompt_data",
                "wrong_response_data",
            ],
            expected_types=["str, list[str]"],
        ),
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "wrong_template",
                "wrong_template_version",
            ],
            expected_types=["string"],
        ),
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "wrong_model_name",
                "wrong_llm_params",
            ],
            expected_types=["string"],
        ),
    ]

    assert len(errors) == len(expected_errors)
    for error, expected_error in zip(errors, expected_errors):
        assert error.error_message() == expected_error.error_message()
        assert type(error) is type(expected_error)

    # prompt type: str
    # response type: str
    schema = kwargs["schema"].replace(
        prompt_column_names="wrong_prompt_data",
        response_column_names="wrong_response_data",
    )
    errors = Validator.validate_types(
        **ChainMap(
            {
                "schema": schema,
                "model_type": ModelTypes.GENERATIVE_LLM,
            },
            kwargs,
        )  # type:ignore
    )
    expected_errors = [
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "wrong_prompt_data",
                "wrong_response_data",
            ],
            expected_types=["str"],
        ),
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "wrong_template",
                "wrong_template_version",
            ],
            expected_types=["string"],
        ),
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "wrong_model_name",
                "wrong_llm_params",
            ],
            expected_types=["string"],
        ),
    ]

    assert len(errors) == len(expected_errors)
    for error, expected_error in zip(errors, expected_errors):
        assert error.error_message() == expected_error.error_message()
        assert type(error) is type(expected_error)


def test_invalid_type_corpus():
    # check valid types
    kwargs = get_corpus_kwargs()
    errors = Validator.validate_types(
        **kwargs,
    )
    assert len(errors) == 0

    # check wrong types
    kwargs = get_corpus_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {
                            "document_id": pd.Series([1.1 for x in range(3)]),
                            "document_version": pd.Series(
                                [1.1 for x in range(3)]
                            ),
                            "document_vector": ["abc" for x in range(3)],
                            "document_data": [x for x in range(3)],
                        }
                    )
                ),
            },
            kwargs,
        )  # type:ignore
    )
    expected_errors = [
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "document_id",
            ],
            expected_types=["str", "int"],
        ),
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "document_version",
            ],
            expected_types=["str"],
        ),
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "document_vector",
            ],
            expected_types=["list[float], np.array[float]"],
        ),
        err.InvalidTypeColumns(
            wrong_type_columns=[
                "document_data",
            ],
            expected_types=["list[str]"],
        ),
    ]

    assert len(errors) == len(expected_errors)
    for error, expected_error in zip(errors, expected_errors):
        assert error.error_message() == expected_error.error_message()
        assert type(error) is type(expected_error)


def test_invalid_type_retrieved_document_ids():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {"retrieved_document_ids": pd.Series(["a", "b"])}
                    )
                )
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidType


def test_invalid_multi_class_score_map():
    kwargs = get_kwargs()
    errors = Validator.validate_types(
        **ChainMap(
            {
                "model_type": ModelTypes.MULTI_CLASS,
            },
            kwargs,
        )  # type:ignore
    )
    assert len(errors) == 3  # 1 row of wrong type
    for error in errors:
        assert type(error) is err.InvalidType

    # dictionary of None should fail
    errors = Validator.validate_types(
        **ChainMap(
            {
                "model_type": ModelTypes.MULTI_CLASS,
                "pyarrow_schema": pa.Schema.from_pandas(
                    pd.DataFrame(
                        {
                            "multi_class_threshold_scores": pd.Series(
                                [
                                    [{}],
                                    [{}],
                                    [{}],
                                ]
                            )
                        }
                    )
                ),
            },
            kwargs,
        )  # type:ignore
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidType


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
            feature_column_names=list("ABCDEFGH"),
            tag_column_names=list("ABCDEFG"),
            shap_values_column_names=dict(zip("ABCDEF", "abcdef")),
            multi_class_threshold_scores_column_name="multi_class_threshold_scores",
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
            prompt_template_column_names=PromptTemplateColumnNames(
                template_column_name="wrong_template",
                template_version_column_name="wrong_template_version",
            ),
            llm_config_column_names=LLMConfigColumnNames(
                model_column_name="wrong_model_name",
                params_column_name="wrong_llm_params",
            ),
            retrieved_document_ids_column_name="retrieved_document_ids",
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
                    "prediction_label": pd.Series(
                        ["fraud", "not fraud", "fraud"]
                    ),
                    "prediction_score": pd.Series([0.2, 0.3, 0.4]),
                    "actual_label": pd.Series(
                        ["not fraud", "fraud", "not fraud"]
                    ),
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
                    #####
                    "retrieved_document_ids": pd.Series(
                        [["id1", "id2"], ["id3", "id4"], ["id5"]]
                    ),
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
                    "wrong_prompt_vector": np.random.randn(3, 3)
                    .astype(str)
                    .tolist(),
                    "wrong_response_vector": np.random.choice(
                        a=[True, False], size=(3, 3)
                    ).tolist(),
                    "wrong_prompt_data": [x for x in range(3)],
                    "wrong_response_data": [x for x in range(3)],
                    "wrong_template": [x for x in range(3)],
                    "wrong_template_version": [x for x in range(3)],
                    "wrong_model_name": [x for x in range(3)],
                    "wrong_llm_params": [x for x in range(3)],
                    "multi_class_threshold_scores": [
                        [{"class_name": "dog", "score": "wrong type"}],
                        [{"class_name": "cat", "score": "wrong type"}],
                        [{"class_name": "fish", "score": "wrong type"}],
                    ],
                }
            )
        ),
    }


def get_corpus_kwargs():
    return {
        "model_type": ModelTypes.GENERATIVE_LLM,
        "pyarrow_schema": pa.Schema.from_pandas(
            pd.DataFrame(
                {
                    "document_id": pd.Series(["id" + str(x) for x in range(3)]),
                    "document_version": [
                        "Version {x}" + str(x) for x in range(3)
                    ],
                    "document_vector": [np.random.randn(15) for x in range(3)],
                    "document_data": ["data_" + str(x) for x in range(3)],
                }
            ),
        ),
        "schema": CorpusSchema(
            document_id_column_name="document_id",
            document_version_column_name="document_version",
            document_text_embedding_column_names=EmbeddingColumnNames(
                vector_column_name="document_vector",
                data_column_name="document_data",
            ),
        ),
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
