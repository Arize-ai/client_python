from collections import ChainMap

import arize.pandas.validation.errors as err
import numpy as np
import pandas as pd
import pytest
from arize.pandas.validation.validator import Validator
from arize.utils.constants import MAX_NUMBER_OF_EMBEDDINGS
from arize.utils.types import (
    CorpusSchema,
    EmbeddingColumnNames,
    Environments,
    LLMConfigColumnNames,
    ModelTypes,
    ObjectDetectionColumnNames,
    PromptTemplateColumnNames,
    Schema,
)

EMBEDDING_SIZE = 15


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
                #####
                "A": pd.Series([0]),
                "B": pd.Series([0.0]),
                #####
                "a": pd.Series([0]),
                "b": pd.Series([0.0]),
                "image_vector": np.random.randn(1, 1).tolist(),
                "image_link": ["link_1"],
                "prompt_vector": np.random.randn(1, 1).tolist(),
                "prompt_data": ["This is a test prompt"],
                "response_vector": np.random.randn(1, 1).tolist(),
                "response_data": ["This is a test response"],
            }
        ),
        "schema": Schema(
            prediction_id_column_name="prediction_id",
            prediction_label_column_name="prediction_label",
            actual_label_column_name="actual_label",
            prediction_score_column_name="prediction_score",
            shap_values_column_names=dict(zip("AB", "ab")),
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


def get_corpus_kwargs():
    return {
        "model_id": "corpus_model",
        "model_type": ModelTypes.GENERATIVE_LLM,
        "environment": Environments.CORPUS,
        "dataframe": pd.DataFrame(
            {
                "document_id": pd.Series(["id" + str(x) for x in range(3)]),
                "document_version": ["Version {x}" + str(x) for x in range(3)],
                "document_vector": [np.random.randn(EMBEDDING_SIZE) for x in range(3)],
                "document_data": ["data_" + str(x) for x in range(3)],
            }
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


def test_zero_error():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(**kwargs)
    assert len(errors) == 0


def test_invalid_model_id():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(**ChainMap({"model_id": " "}, kwargs))  # type: ignore
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidModelId


def test_invalid_model_type():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(**ChainMap({"model_type": -1}, kwargs))  # type: ignore
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidModelType


def test_invalid_environment():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(**ChainMap({"environment": -1}, kwargs))  # type: ignore
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidEnvironment


def test_multiple():
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(
        **ChainMap({"model_id": " ", "model_type": -1, "environment": -1}, kwargs)  # type: ignore
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
        )  # type: ignore
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
        )  # type: ignore
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
        )  # type: ignore
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
        )  # type: ignore
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
        )  # type: ignore
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
        )  # type: ignore
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
        )  # type: ignore
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
        )  # type: ignore
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
        )  # type: ignore
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
        )  # type: ignore
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingPredActShapNumericAndCategorical


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
        )  # type: ignore
    )
    assert len(errors) == 0


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
        )  # type: ignore
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingPreprodPredActNumericAndCategorical


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
        )  # type: ignore
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
        )  # type: ignore
    )
    assert len(errors) == 2
    assert any(type(e) is err.MissingPreprodPredActNumericAndCategorical for e in errors)
    assert any(type(e) is err.MissingPredActShapNumericAndCategorical for e in errors)


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
        )  # type: ignore
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingPreprodPredActNumericAndCategorical


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
        )  # type: ignore
    )
    assert len(errors) == 3
    assert any(type(e) is err.InvalidBatchId for e in errors)
    assert any(type(e) is err.MissingPreprodPredActNumericAndCategorical for e in errors)
    assert any(type(e) is err.MissingPredActShapNumericAndCategorical for e in errors)


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
        )  # type: ignore
    )
    assert len(errors) == 0


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
        )  # type: ignore
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
        )  # type: ignore
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
        )  # type: ignore
    )
    assert len(errors) == 1
    for error in errors:
        assert type(error) == err.InvalidPredActColumnNamesForModelType


def test_existence_pred_act_multi_class_column_names():
    kwargs = get_standard_kwargs()
    # Test that if Multi Class models do not contain the prediction/actual column names
    # a missing required columns error is returned
    # It is equivalent to test_missing_columns but for multi class.
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.MULTI_CLASS,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                ),
            },
            kwargs,
        )  # type: ignore
    )
    assert len(errors) == 1
    for error in errors:
        assert type(error) == err.MissingReqPredActColumnNamesForMultiClass


def test_non_multi_class_model_do_not_contain_multi_class_column_names():
    kwargs = get_standard_kwargs()
    # Test that a score categorical (any non-multi-class) model rejects multi clas prediction/actual
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
                    actual_score_column_name="actual_score",
                    multi_class_threshold_scores_column_name="prediction_score",
                ),
                "dataframe": pd.DataFrame(
                    {
                        "prediction_id": pd.Series(["0"]),
                        "prediction_label": pd.Series(["fraud"]),
                        "prediction_score": pd.Series([1]),
                        "actual_label": pd.Series(["not fraud"]),
                        "actual_score": pd.Series([0]),
                        "multi_class_threshold_scores": pd.Series(
                            [
                                {"cat": 0.1, "dog": 0.3},
                            ]
                        ),
                    }
                ),
            },
            kwargs,
        )  # type: ignore
    )
    assert len(errors) == 1
    for error in errors:
        assert type(error) == err.InvalidPredActColumnNamesForModelType


def test_duplicate_column_names_in_dataframe():
    kwargs = get_standard_kwargs()
    # We add a duplicate "prediction_score" column
    df_with_duplicate_column = pd.concat(
        [kwargs["dataframe"], pd.DataFrame({"prediction_score": pd.Series([1])})], axis=1
    )

    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": df_with_duplicate_column,
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.DuplicateColumnsInDataframe)
    # We add a duplicate "A" column that is used as a feature
    df_with_duplicate_feature_column = pd.concat(
        [kwargs["dataframe"], pd.DataFrame({"A": pd.Series([2])})], axis=1
    )

    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": df_with_duplicate_feature_column,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_score_column_name="prediction_score",
                    feature_column_names=list("A"),
                    prediction_label_column_name="prediction_label",
                    actual_label_column_name="actual_label",
                ),
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.DuplicateColumnsInDataframe)
    # We add a duplicate "image_vector" column used in embedding_feature_column
    df_with_duplicate_embedding_feature_column = pd.concat(
        [
            kwargs["dataframe"],
            pd.DataFrame({"image_vector": np.random.randn(1, EMBEDDING_SIZE).tolist()}),
        ],
        axis=1,
    )

    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": df_with_duplicate_embedding_feature_column,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    actual_label_column_name="actual_label",
                    embedding_feature_column_names={
                        "image_embedding": EmbeddingColumnNames(
                            vector_column_name="image_vector",
                            link_to_data_column_name="image_link",
                        ),
                    },
                ),
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.DuplicateColumnsInDataframe)


def test_existence_prediction_id_column():
    kwargs = get_standard_kwargs()
    # Test case - prediction_id_column_name not None, not latent info
    errors = Validator.validate_params(**kwargs)
    assert len(errors) == 0
    # Test case - prediction_id_column_name not None, latent info
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    actual_label_column_name="actual_label",
                ),
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 0
    # Test case - prediction_id_column_name None, not latent info
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_label_column_name="prediction_label",
                    actual_label_column_name="actual_label",
                ),
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 0
    # Test case - prediction_id_column_name None, latent actuals
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    actual_label_column_name="actual_label",
                ),
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingPredictionIdColumnForDelayedRecords
    # Test case - prediction_id_column_name None, latent shap
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": Schema(
                    shap_values_column_names=dict(zip("AB", "ab")),
                ),
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.MissingPredictionIdColumnForDelayedRecords
    # Test case - prediction_id_column_name None, latent actual, generative model type
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.GENERATIVE_LLM,
                "schema": Schema(
                    actual_label_column_name="actual_label",
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
        ),  # type: ignore
    )
    assert len(errors) == 0


def test_missing_prompt_templates_and_llm_config_columns():
    kwargs = get_standard_kwargs()
    schema = kwargs["schema"].replace(
        prompt_template_column_names=PromptTemplateColumnNames(
            template_column_name="prompt_templates",
            template_version_column_name="prompt_template_version",
        ),
        llm_config_column_names=LLMConfigColumnNames(
            model_column_name="llm_model_name",
            params_column_name="llm_params",
        ),
    )
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": schema,
            },
            kwargs,
        )  # type: ignore
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.MissingColumns)

    dataframe = kwargs["dataframe"]
    dataframe["prompt_templates"] = ["This is the template with version {{version}}"]
    dataframe["prompt_template_version"] = ["Template A"]
    dataframe["llm_model_name"] = ["gpt-3.5turbo"]
    dataframe["llm_params"] = [
        {"temperature": 1 / 4, "presence_penalty": 1 / 3, "stop": [".", "?", "!"]}
    ]
    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": dataframe,
                "schema": schema,
            },
            kwargs,
        )  # type: ignore
    )
    assert len(errors) == 0


def test_missing_document_columns():
    # test valid
    kwargs = get_corpus_kwargs()
    errors = Validator.validate_params(**kwargs)
    assert len(errors) == 0

    # missing document id column
    kwargs = get_corpus_kwargs()
    schema = kwargs["schema"].replace(document_id_column_name="nonexistent_column")
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": schema,
            },
            kwargs,
        )  # type: ignore
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.MissingColumns)

    # missing document version column
    kwargs = get_corpus_kwargs()
    schema = kwargs["schema"].replace(document_version_column_name="nonexistent_column")
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": schema,
            },
            kwargs,
        )  # type: ignore
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.MissingColumns)

    # missing document embedding column
    kwargs = get_corpus_kwargs()
    schema = kwargs["schema"].replace(
        document_text_embedding_column_names=EmbeddingColumnNames(
            vector_column_name="nonexistent_column",
            data_column_name="nonexistent_column",
        ),
    )
    errors = Validator.validate_params(
        **ChainMap(
            {
                "schema": schema,
            },
            kwargs,
        )  # type: ignore
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.MissingColumns)


def test_missing_and_incorrect_multi_class_columns():
    # test valid
    kwargs = get_standard_kwargs()
    errors = Validator.validate_params(**kwargs)
    assert len(errors) == 0

    # missing document id column
    schema = kwargs["schema"].replace(multi_class_threshold_scores_column_name="nonexistent_column")
    errors = Validator.validate_params(
        **ChainMap(
            {
                "model_type": ModelTypes.MULTI_CLASS,
                "schema": schema,
            },
            kwargs,
        )  # type: ignore
    )
    assert len(errors) == 2
    for error in errors:
        assert isinstance(
            error,
            (
                err.MissingColumns,
                err.InvalidPredActColumnNamesForModelType,
            ),
        )


def test_invalid_number_of_embeddings():
    kwargs = get_standard_kwargs()
    schema = kwargs["schema"]
    # Testing success
    embedding_features = {
        f"embedding_feat_{i:02d}": EmbeddingColumnNames(
            vector_column_name="image_vector",
            link_to_data_column_name="image_link",
        )
        for i in range(MAX_NUMBER_OF_EMBEDDINGS)
    }
    schema = schema.replace(
        embedding_feature_column_names=embedding_features,
    )
    errors = Validator.validate_params(
        **ChainMap(
            {"schema": schema},
            kwargs,
        )  # type: ignore
    )
    assert len(errors) == 0
    # Testing error
    embedding_features = {
        f"embedding_feat_{i:02d}": EmbeddingColumnNames(
            vector_column_name="image_vector",
            link_to_data_column_name="image_link",
        )
        for i in range(MAX_NUMBER_OF_EMBEDDINGS + 1)
    }
    schema = schema.replace(
        embedding_feature_column_names=embedding_features,
    )
    errors = Validator.validate_params(
        **ChainMap(
            {"schema": schema},
            kwargs,
        )  # type: ignore
    )
    assert len(errors) == 1
    assert type(errors[0]) is err.InvalidNumberOfEmbeddings


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
