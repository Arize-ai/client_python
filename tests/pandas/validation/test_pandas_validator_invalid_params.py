from collections import ChainMap

import arize.pandas.validation.errors as err
import pandas as pd
import pytest
from arize.pandas.logger import Schema
from arize.pandas.validation.validator import Validator
from arize.utils.types import (
    EmbeddingColumnNames,
    Environments,
    ModelTypes,
    ObjectDetectionColumnNames,
)


def get_standard_kwargs():
    return {
        "model_id": "fraud",
        "model_type": ModelTypes.SCORE_CATEGORICAL,
        "environment": Environments.PRODUCTION,
        "dataframe": pd.DataFrame(
            {
                "prediction_id": pd.Series(["0"]),
                "prediction_label": pd.Series(["fraud"]),
                "prediction_score": pd.Series([1]),
                "actual_label": pd.Series(["not fraud"]),
                "actual_score": pd.Series([0]),
            }
        ),
        "schema": Schema(
            prediction_id_column_name="prediction_id",
            prediction_label_column_name="prediction_label",
            actual_label_column_name="actual_label",
            prediction_score_column_name="prediction_score",
        ),
    }


def get_object_detection_kwargs():
    return {
        "model_id": "fraud",
        "model_type": ModelTypes.OBJECT_DETECTION,
        "environment": Environments.PRODUCTION,
        "dataframe": pd.DataFrame(
            {
                "prediction_id": pd.Series(["0"]),
                "bounding_boxes_coordinates": pd.Series(
                    [
                        [[0.31, 0.32, 0.33, 0.34], [0.31, 0.32, 0.33, 0.34]],
                    ]
                ),
                "bounding_boxes_categories": pd.Series(
                    [
                        ["dog", "cat"],
                    ]
                ),
                "bounding_boxes_scores": pd.Series(
                    [
                        [0.18, 0.33],
                    ]
                ),
            }
        ),
        "schema": Schema(
            prediction_id_column_name="prediction_id",
            object_detection_prediction_column_names=ObjectDetectionColumnNames(
                bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
                categories_column_name="bounding_boxes_categories",
                scores_column_name="bounding_boxes_scores",
            ),
            object_detection_actual_column_names=ObjectDetectionColumnNames(
                bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
                categories_column_name="bounding_boxes_categories",
                scores_column_name="bounding_boxes_scores",
            ),
        ),
    }


def test_zero_error():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(**kwargs)
    assert len(errors) == 0


def test_invalid_model_id():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(**ChainMap({"model_id": " "}, kwargs))
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidModelId


def test_invalid_model_type():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(**ChainMap({"model_type": -1}, kwargs))
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidModelType


def test_invalid_environment():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(**ChainMap({"environment": -1}, kwargs))
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidEnvironment


def test_multiple():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap({"model_id": " ", "model_type": -1, "environment": -1}, kwargs)
    )
    assert len(errors) == 3
    assert any(type(e) is err.InvalidModelId for e in errors)
    assert any(type(e) is err.InvalidModelType for e in errors)
    assert any(type(e) is err.InvalidEnvironment for e in errors)


def test_invalid_batch_id_none():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": "1.0",
                "environment": Environments.VALIDATION,
                "batch_id": None,
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidBatchId


def test_invalid_batch_id_empty_str():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": "1.0",
                "environment": Environments.VALIDATION,
                "batch_id": "",
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidBatchId


def test_invalid_batch_id_blank_str():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": "1.0",
                "environment": Environments.VALIDATION,
                "batch_id": "  ",
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidBatchId


def test_invalid_model_version_int_train():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": 2,
                "environment": Environments.TRAINING,
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidModelVersion


def test_invalid_model_version_empty_str_train():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": "",
                "environment": Environments.TRAINING,
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidModelVersion


def test_invalid_model_version_blank_str_train():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": "  ",
                "environment": Environments.TRAINING,
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidModelVersion


def test_invalid_model_version_int_val():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": 2,
                "environment": Environments.VALIDATION,
                "batch_id": "1",
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidModelVersion


def test_invalid_model_version_empty_str_val():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": "",
                "environment": Environments.VALIDATION,
                "batch_id": "1",
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidModelVersion


def test_invalid_model_version_blank_str_val():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": "  ",
                "environment": Environments.VALIDATION,
                "batch_id": "1",
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidModelVersion


def test_missing_pred_act_shap():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "environment": Environments.PRODUCTION,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingPredActShap


def test_missing_pred_label_score_categorical():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": "1.0",
                "environment": Environments.PRODUCTION,
                "model_type": ModelTypes.SCORE_CATEGORICAL,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_score_column_name="prediction_score",
                    actual_label_column_name="actual_label",
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingPredLabelScoreCategorical


def test_missing_preprod_pred_act_train():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": "1.0",
                "environment": Environments.TRAINING,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingPreprodPredAct


def test_missing_preprod_act_train():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": "1.0",
                "model_type": ModelTypes.GENERATIVE_LLM,
                "environment": Environments.TRAINING,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prompt_column_names=EmbeddingColumnNames(
                        vector_column_name="prompt_vector",
                        data_column_name="prompt_data",
                    ),
                    response_column_names=EmbeddingColumnNames(
                        vector_column_name="response_vector",
                        data_column_name="response_data",
                    ),
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingPreprodAct


def test_missing_multiple_train():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "environment": Environments.TRAINING,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 2
    assert any(type(e) is err.MissingPreprodPredAct for e in errors)
    assert any(type(e) is err.MissingPredActShap for e in errors)


def test_missing_preprod_pred_act_val():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_version": "1.0",
                "environment": Environments.VALIDATION,
                "batch_id": "1",
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingPreprodPredAct


def test_missing_multiple_val():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap(
            {
                "environment": Environments.VALIDATION,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 3
    assert any(type(e) is err.InvalidBatchId for e in errors)
    assert any(type(e) is err.MissingPreprodPredAct for e in errors)
    assert any(type(e) is err.MissingPredActShap for e in errors)


def test_existence_prompt_response_column_names():
    kwargs = get_standard_kwargs()
    # Test that Generative LLM models contain the prompt/response column names
    # It is equivalent to test_missing_columns but for Generative LLM.
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.GENERATIVE_LLM,
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingPromptResponseGenerativeLLM


def test_existence_pred_act_od_column_names():
    object_detection_kwargs = get_object_detection_kwargs()
    # Test that Object Detection models contain the prediction/actual column names
    # It is equivalent to test_missing_columns but for object detection.
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.OBJECT_DETECTION,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                ),
            },
            object_detection_kwargs,
        )
    )
    assert len(errors) == 1
    for error in errors:
        assert type(error) == err.MissingObjectDetectionPredAct


def test_non_existence_pred_act_od_column_names():
    kwargs = get_standard_kwargs()
    # Test that a non-object-detection model should not have object_detection prediction/actual
    # column names in the schema
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.SCORE_CATEGORICAL,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    actual_label_column_name="actual_label",
                    prediction_score_column_name="prediction_score",
                    object_detection_prediction_column_names=ObjectDetectionColumnNames(
                        bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
                        categories_column_name="bounding_boxes_categories",
                        scores_column_name="bounding_boxes_scores",
                    ),
                    object_detection_actual_column_names=ObjectDetectionColumnNames(
                        bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
                        categories_column_name="bounding_boxes_categories",
                        scores_column_name="bounding_boxes_scores",
                    ),
                ),
                "dataframe": pd.DataFrame(
                    {
                        "prediction_id": pd.Series(["0"]),
                        "prediction_label": pd.Series(["fraud"]),
                        "prediction_score": pd.Series([1]),
                        "actual_label": pd.Series(["not fraud"]),
                        "actual_score": pd.Series([0]),
                        "bounding_boxes_coordinates": pd.Series(
                            [
                                [[0.31, 0.32, 0.33, 0.34], [0.31, 0.32, 0.33, 0.34]],
                            ]
                        ),
                        "bounding_boxes_categories": pd.Series(
                            [
                                ["dog", "cat"],
                            ]
                        ),
                        "bounding_boxes_scores": pd.Series(
                            [
                                [0.18, 0.33],
                            ]
                        ),
                    }
                ),
            },
            kwargs,
        )
    )
    assert len(errors) == 1
    for error in errors:
        assert type(error) == err.InvalidPredActObjectDetectionColumnNamesForModelType


def test_non_existence_pred_act_column_name():
    object_detection_kwargs = get_object_detection_kwargs()
    # Check that object detection should not have the prediction/actual column names reserved for
    # other model types
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.OBJECT_DETECTION,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    actual_label_column_name="actual_label",
                    prediction_score_column_name="prediction_score",
                    object_detection_prediction_column_names=ObjectDetectionColumnNames(
                        bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
                        categories_column_name="bounding_boxes_categories",
                        scores_column_name="bounding_boxes_scores",
                    ),
                    object_detection_actual_column_names=ObjectDetectionColumnNames(
                        bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
                        categories_column_name="bounding_boxes_categories",
                        scores_column_name="bounding_boxes_scores",
                    ),
                ),
                "dataframe": pd.DataFrame(
                    {
                        "prediction_id": pd.Series(["0"]),
                        "prediction_label": pd.Series(["fraud"]),
                        "prediction_score": pd.Series([1]),
                        "actual_label": pd.Series(["not fraud"]),
                        "actual_score": pd.Series([0]),
                        "bounding_boxes_coordinates": pd.Series(
                            [
                                [[0.31, 0.32, 0.33, 0.34], [0.31, 0.32, 0.33, 0.34]],
                            ]
                        ),
                        "bounding_boxes_categories": pd.Series(
                            [
                                ["dog", "cat"],
                            ]
                        ),
                        "bounding_boxes_scores": pd.Series(
                            [
                                [0.18, 0.33],
                            ]
                        ),
                    }
                ),
            },
            object_detection_kwargs,
        )
    )
    assert len(errors) == 1
    for error in errors:
        assert type(error) == err.InvalidPredActColumnNamesForObjectDetectionModelType


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
