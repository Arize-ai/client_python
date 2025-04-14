import datetime
import sys
import uuid
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from attr import dataclass
from requests import Response

import arize.pandas.validation.errors as err
from arize.pandas.etl.errors import ColumnCastingError, InvalidTypedColumnsError
from arize.pandas.logger import Client
from arize.utils.constants import (
    GENERATED_LLM_PARAMS_JSON_COL,
    GENERATED_PREDICTION_LABEL_COL,
    LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
    MAX_EMBEDDING_DIMENSIONALITY,
    MAX_LLM_MODEL_NAME_LENGTH,
    MAX_NUMBER_OF_MULTI_CLASS_CLASSES,
    MAX_PROMPT_TEMPLATE_LENGTH,
    MAX_PROMPT_TEMPLATE_VERSION_LENGTH,
    MAX_RAW_DATA_CHARACTERS,
)
from arize.utils.errors import AuthError
from arize.utils.types import (
    CorpusSchema,
    EmbeddingColumnNames,
    Environments,
    InstanceSegmentationActualColumnNames,
    InstanceSegmentationPredictionColumnNames,
    LLMConfigColumnNames,
    LLMRunMetadataColumnNames,
    ModelTypes,
    ObjectDetectionColumnNames,
    PromptTemplateColumnNames,
    Schema,
    SemanticSegmentationColumnNames,
    TypedColumns,
)
from arize.utils.utils import get_python_version
from arize.version import __version__ as arize_version

EMBEDDING_SIZE = 15


class MockResponse(Response):
    def __init__(self, df, reason, status_code):
        super().__init__()
        self.df = df
        self.reason = reason
        self.status_code = status_code


class NoSendClient(Client):
    def _post_file(self, path, sync, timeout):
        return MockResponse(
            pa.ipc.open_stream(pa.OSFile(path)).read_pandas(), "Success", 200
        )


def get_base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prediction_id": pd.Series([str(uuid.uuid4()) for _ in range(3)]),
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
            "D": pd.Series([0, float("NaN"), 2]),
            "E": pd.Series([0, None, 2]),
            "F": pd.Series([None, float("NaN"), None]),
            "G": pd.Series(["foo", "bar", "baz"]),
            "H": pd.Series([True, False, True]),
            "I": pd.Categorical(
                ["a", "b", "c"], ordered=True, categories=["c", "b", "a"]
            ),
            #####
            "a": pd.Series([0, 1, 2]),
            "b": pd.Series([0.0, 1.0, 2.0]),
            "c": pd.Series([float("NaN"), float("NaN"), float("NaN")]),
            "d": pd.Series([0, float("NaN"), 2]),
            "e": pd.Series([0, None, 2]),
            "f": pd.Series([None, float("NaN"), None]),
            #####
            "excluded_from_schema": pd.Series(
                [
                    "should also be excluded from pyarrow",
                    0,
                    "otherwise would cause error (because of mixed types)",
                ]
            ),
        }
    )


def get_base_schema() -> Schema:
    return Schema(
        prediction_id_column_name="prediction_id",
        timestamp_column_name="prediction_timestamp",
        feature_column_names=list("ABCDEFGHI"),
        tag_column_names=list("ABCDEFGHI"),
        shap_values_column_names=dict(zip("ABCDEF", "abcdef")),
    )


def get_score_categorical_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prediction_label": pd.Series(["fraud", "not fraud", "fraud"]),
            "prediction_score": pd.Series([0.2, 0.3, 0.4]),
            "actual_label": pd.Categorical(
                ["not fraud", "fraud", "not fraud"],
                ordered=True,
                categories=["fraud", "not fraud"],
            ),
            "actual_score": pd.Series([0, 1, 0]),
        }
    )


def get_score_categorical_schema() -> Schema:
    return Schema(
        prediction_label_column_name="prediction_label",
        actual_label_column_name="actual_label",
        prediction_score_column_name="prediction_score",
        actual_score_column_name="actual_score",
    )


def get_embeddings_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_vector": np.random.randn(3, EMBEDDING_SIZE).tolist(),
            "image_link": ["link_" + str(x) for x in range(3)],
            "sentence_vector": [
                np.random.randn(EMBEDDING_SIZE) for x in range(3)
            ],
            "sentence_data": ["data_" + str(x) for x in range(3)],
            "token_array_vector": [
                np.random.randn(EMBEDDING_SIZE) for x in range(3)
            ],
            "token_array_data": [["Token", "array", str(x)] for x in range(3)],
        }
    )


def get_embeddings_schema() -> Schema:
    return Schema(
        embedding_feature_column_names={
            # Dictionary keys will be the displayed name of the embedding feature in the app
            "image_embedding": EmbeddingColumnNames(
                vector_column_name="image_vector",
                link_to_data_column_name="image_link",
            ),
            "sentence_embedding": EmbeddingColumnNames(
                vector_column_name="sentence_vector",
                data_column_name="sentence_data",
            ),
            "token_embedding": EmbeddingColumnNames(
                vector_column_name="token_array_vector",
                data_column_name="token_array_data",
            ),
        },
    )


def get_object_detection_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_vector": np.random.randn(3, EMBEDDING_SIZE).tolist(),
            "image_link": ["link_" + str(x) for x in range(3)],
            "bounding_boxes_coordinates": pd.Series(
                [
                    [[0.11, 0.12, 0.13, 0.14], [0.11, 0.12, 0.13, 0.14]],
                    [[0.21, 0.22, 0.23, 0.24], [0.21, 0.22, 0.23, 0.24]],
                    [[0.31, 0.32, 0.33, 0.34], [0.31, 0.32, 0.33, 0.34]],
                ]
            ),
            "bounding_boxes_categories": pd.Series(
                [
                    ["dog", "cat"],
                    ["lion", "tiger"],
                    ["elephant", "butterfly"],
                ]
            ),
            "bounding_boxes_scores": pd.Series(
                [
                    [0.18, 0.33],
                    [0.28, 0.23],
                    [0.38, 0.13],
                ]
            ),
        }
    )


def get_object_detection_schema() -> Schema:
    return Schema(
        embedding_feature_column_names={
            # Dictionary keys will be the displayed name of the embedding feature in the app
            "image_embedding": EmbeddingColumnNames(
                vector_column_name="image_vector",
                link_to_data_column_name="image_link",
            ),
        },
        object_detection_prediction_column_names=ObjectDetectionColumnNames(
            bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
            categories_column_name="bounding_boxes_categories",
            scores_column_name="bounding_boxes_scores",
        ),
        object_detection_actual_column_names=ObjectDetectionColumnNames(
            bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
            categories_column_name="bounding_boxes_categories",
        ),
    )


def get_semantic_segmentation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_vector": np.random.randn(3, EMBEDDING_SIZE).tolist(),
            "image_link": ["link_" + str(x) for x in range(3)],
            "polygon_coordinates": pd.Series(
                [
                    [
                        [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1],
                        [0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.2],
                    ],
                    [
                        [0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.3],
                        [0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4],
                    ],
                    [
                        [0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5],
                        [0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6],
                    ],
                ]
            ),
            "polygon_categories": pd.Series(
                [
                    ["dog", "cat"],
                    ["lion", "tiger"],
                    ["elephant", "butterfly"],
                ]
            ),
        }
    )


def get_semantic_segmentation_schema() -> Schema:
    return Schema(
        embedding_feature_column_names={
            # Dictionary keys will be the displayed name of the embedding feature in the app
            "image_embedding": EmbeddingColumnNames(
                vector_column_name="image_vector",
                link_to_data_column_name="image_link",
            ),
        },
        semantic_segmentation_prediction_column_names=SemanticSegmentationColumnNames(
            polygon_coordinates_column_name="polygon_coordinates",
            categories_column_name="polygon_categories",
        ),
        semantic_segmentation_actual_column_names=SemanticSegmentationColumnNames(
            polygon_coordinates_column_name="polygon_coordinates",
            categories_column_name="polygon_categories",
        ),
    )


def get_instance_segmentation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_vector": np.random.randn(3, EMBEDDING_SIZE).tolist(),
            "image_link": ["link_" + str(x) for x in range(3)],
            "polygon_coordinates": pd.Series(
                [
                    [
                        [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1],
                        [0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.2],
                    ],
                    [
                        [0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.3],
                        [0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.4],
                    ],
                    [
                        [0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5],
                        [0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6],
                    ],
                ]
            ),
            "polygon_categories": pd.Series(
                [
                    ["dog", "cat"],
                    ["lion", "tiger"],
                    ["elephant", "butterfly"],
                ]
            ),
            "polygon_scores": pd.Series(
                [
                    [0.18, 0.33],
                    [0.28, 0.23],
                    [0.38, 0.13],
                ]
            ),
            "bounding_boxes_coordinates": pd.Series(
                [
                    [[0.1, 0.1, 0.2, 0.2], [0.2, 0.2, 0.3, 0.3]],
                    [[0.3, 0.3, 0.4, 0.4], [0.4, 0.4, 0.5, 0.5]],
                    [[0.5, 0.5, 0.6, 0.6], [0.6, 0.6, 0.7, 0.7]],
                ]
            ),
        }
    )


def get_instance_segmentation_schema() -> Schema:
    return Schema(
        embedding_feature_column_names={
            # Dictionary keys will be the displayed name of the embedding feature in the app
            "image_embedding": EmbeddingColumnNames(
                vector_column_name="image_vector",
                link_to_data_column_name="image_link",
            ),
        },
        instance_segmentation_prediction_column_names=InstanceSegmentationPredictionColumnNames(
            polygon_coordinates_column_name="polygon_coordinates",
            categories_column_name="polygon_categories",
            scores_column_name="polygon_scores",
            bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
        ),
        instance_segmentation_actual_column_names=InstanceSegmentationActualColumnNames(
            polygon_coordinates_column_name="polygon_coordinates",
            categories_column_name="polygon_categories",
            bounding_boxes_coordinates_column_name="bounding_boxes_coordinates",
        ),
    )


def get_generative_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt_vector": np.random.randn(3, EMBEDDING_SIZE).tolist(),
            "response_vector": np.random.randn(3, EMBEDDING_SIZE).tolist(),
            "prompt_data": ["data_" + str(x) for x in range(3)],
            "response_data": ["data_" + str(x) for x in range(3)],
            "llm_model_name": ["gpt-" + str(x) for x in range(3)],
            "prompt_template": [
                "This is the template with version {{version}}"
                for _ in range(3)
            ],
            "prompt_template_version": [
                "Template {x}" + str(x) for x in range(3)
            ],
            "llm_params": [
                {
                    "presence_penalty": x / 3,
                    "stop": [".", "?", "!"],
                    "temperature": x / 4,
                }
                for x in range(3)
            ],
            "total_token_count": [x * 30 for x in range(3)],
            "prompt_token_count": [x * 20 for x in range(3)],
            "response_token_count": [x * 10 for x in range(3)],
            "response_latency_ms": [x * 1000 for x in range(3)],
            "retrieved_document_ids": [
                [str(x * 3 + i) for i in range(3)] for x in range(3)
            ],
        }
    )


def get_generative_schema() -> Schema:
    return Schema(
        prompt_column_names=EmbeddingColumnNames(
            vector_column_name="prompt_vector",
            data_column_name="prompt_data",
        ),
        response_column_names=EmbeddingColumnNames(
            vector_column_name="response_vector",
            data_column_name="response_data",
        ),
        prompt_template_column_names=PromptTemplateColumnNames(
            template_column_name="prompt_template",
            template_version_column_name="prompt_template_version",
        ),
        llm_config_column_names=LLMConfigColumnNames(
            model_column_name="llm_model_name",
            params_column_name="llm_params",
        ),
        llm_run_metadata_column_names=LLMRunMetadataColumnNames(
            total_token_count_column_name="total_token_count",
            prompt_token_count_column_name="prompt_token_count",
            response_token_count_column_name="response_token_count",
            response_latency_ms_column_name="response_latency_ms",
        ),
        retrieved_document_ids_column_name="retrieved_document_ids",
    )


def get_corpus_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "document_id": pd.Series([str(uuid.uuid4()) for _ in range(3)]),
            "document_version": ["Version {x}" + str(x) for x in range(3)],
            "document_vector": [
                np.random.randn(EMBEDDING_SIZE) for x in range(3)
            ],
            "document_data": ["data_" + str(x) for x in range(3)],
        }
    )


def get_corpus_schema() -> Schema:
    return CorpusSchema(
        document_id_column_name="document_id",
        document_version_column_name="document_version",
        document_text_embedding_column_names=EmbeddingColumnNames(
            vector_column_name="document_vector",
            data_column_name="document_data",
        ),
    )


def get_multi_class_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prediction_scores": pd.Series(
                [
                    [
                        {"class_name": "dog", "score": 0.1},
                        {"class_name": "cat", "score": 0.2},
                        {"class_name": "fish", "score": 0.3},
                    ],
                    [
                        {"class_name": "dog", "score": 0.1},
                        {"class_name": "cat", "score": 0.2},
                        {"class_name": "fish", "score": 0.3},
                    ],
                    [
                        {"class_name": "dog", "score": 0.1},
                        {"class_name": "cat", "score": 0.2},
                        {"class_name": "fish", "score": 0.3},
                    ],
                ]
            ),
            "threshold_scores": pd.Series(
                [
                    [
                        {"class_name": "dog", "score": 0.1},
                        {"class_name": "cat", "score": 0.2},
                        {"class_name": "fish", "score": 0.3},
                    ],
                    [
                        {"class_name": "dog", "score": 0.1},
                        {"class_name": "cat", "score": 0.2},
                        {"class_name": "fish", "score": 0.3},
                    ],
                    [
                        {"class_name": "dog", "score": 0.1},
                        {"class_name": "cat", "score": 0.2},
                        {"class_name": "fish", "score": 0.3},
                    ],
                ]
            ),
            "actual_scores": pd.Series(
                [
                    [
                        {"class_name": "dog", "score": 0},
                        {"class_name": "cat", "score": 0},
                        {"class_name": "fish", "score": 1},
                    ],
                    [
                        {"class_name": "dog", "score": 0},
                        {"class_name": "cat", "score": 1},
                        {"class_name": "fish", "score": 0},
                    ],
                    [
                        {"class_name": "dog", "score": 1},
                        {"class_name": "cat", "score": 0},
                        {"class_name": "fish", "score": 0},
                    ],
                ]
            ),
        }
    )


def get_multi_class_schema() -> Schema:
    return Schema(
        prediction_score_column_name="prediction_scores",
        multi_class_threshold_scores_column_name="threshold_scores",
        actual_score_column_name="actual_scores",
    )


# roundtrip df is the expected df that would be re-constructed from the pyarrow serialization, where
# 1. the column excluded from schema should have been dropped
# 2. categorical variables should have been converted to string
def roundtrip_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("excluded_from_schema", axis=1, errors="ignore")
    return df.astype(
        {k: "str" for k, v in df.dtypes.items() if v.name == "category"}
    )


def log_dataframe(
    df: pd.DataFrame,
    schema: Schema,
    model_type: ModelTypes,
    environment=Environments.PRODUCTION,
):
    client = NoSendClient("apikey", "spaceKey")

    response = client.log(
        dataframe=df,
        model_id="model-id",
        model_version="1.0",
        model_type=model_type,
        environment=environment,
        schema=schema,
    )
    return response


def test_zero_errors():
    # CHECK score categorical model with embeddings
    data_df = pd.concat(
        [get_base_df(), get_score_categorical_df(), get_embeddings_df()], axis=1
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    try:
        response = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        pytest.fail("Unexpected error")
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )

    # CHECK logging without prediction_id
    schema = schema.replace(prediction_id_column_name=None)
    try:
        response = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        pytest.fail("Unexpected error")
    data_df = data_df.drop(columns=["prediction_id"])
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )

    # CHECK logging generative model
    data_df = pd.concat(
        [
            get_base_df(),
            get_score_categorical_df(),
            get_embeddings_df(),
            get_generative_df(),
        ],
        axis=1,
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())
    try:
        response = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        pytest.fail("Unexpected error")
    # assert that we generated the reserved llm run metadata tag columns correctly
    assert response_df[LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME].equals(
        data_df["total_token_count"]
    )
    assert response_df[LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME].equals(
        data_df["prompt_token_count"]
    )
    assert response_df[LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME].equals(
        data_df["response_token_count"]
    )
    assert response_df[LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME].equals(
        data_df["response_latency_ms"]
    )
    response_df = response_df.drop(columns=[GENERATED_LLM_PARAMS_JSON_COL])
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )

    # CHECK logging corpus environment
    data_df = pd.concat(
        [get_corpus_df()],
        axis=1,
    )
    schema = get_corpus_schema()
    try:
        response = log_dataframe(
            data_df,
            schema=schema,
            model_type=ModelTypes.GENERATIVE_LLM,
            environment=Environments.CORPUS,
        )
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception as e:
        raise AssertionError("Unexpected error") from e
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_zero_errors_object_detection():
    # CHECK logging object detection model
    data_df = pd.concat([get_base_df(), get_object_detection_df()], axis=1)
    schema = _overwrite_schema_fields(
        get_base_schema(), get_object_detection_schema()
    )
    try:
        response = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.OBJECT_DETECTION
        )
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        pytest.fail("Unexpected error")
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_zero_errors_semantic_segmentation():
    # CHECK logging object detection model with semantic segmentation columns
    data_df = pd.concat([get_base_df(), get_semantic_segmentation_df()], axis=1)
    schema = _overwrite_schema_fields(
        get_base_schema(), get_semantic_segmentation_schema()
    )
    try:
        response = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.OBJECT_DETECTION
        )
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        pytest.fail("Unexpected error")
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_zero_errors_instance_segmentation():
    # CHECK logging object detection model with instance segmentation columns
    data_df = pd.concat([get_base_df(), get_instance_segmentation_df()], axis=1)
    schema = _overwrite_schema_fields(
        get_base_schema(), get_instance_segmentation_schema()
    )
    try:
        response = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.OBJECT_DETECTION
        )
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        pytest.fail("Unexpected error")
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


def test_zero_errors_multi_class():
    # CHECK logging multi class model
    data_df = pd.concat(
        [get_base_df(), get_multi_class_df()],
        axis=1,
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_multi_class_schema()
    )
    try:
        response = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.MULTI_CLASS
        )
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        pytest.fail("Unexpected error")
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


def test_invalid_multi_class():
    # CHECK logging multi class model
    data_df = pd.concat(
        [get_base_df(), get_multi_class_df()],
        axis=1,
    )
    data_df["prediction_scores"] = pd.Series(
        [
            [
                {"class_name": "dog", "score": None},  # score can't be None
                {"class_name": "cat", "score": 0.2},
                {"class_name": "fish", "score": 0.3},
            ],
            [
                {"class_name": "dog", "score": 0.1},
                {"class_name": "cat", "score": 0.2},
                {"class_name": "fish", "score": 0.3},
            ],
            [
                {"class_name": "dog", "score": 0.1},
                {"class_name": "cat", "score": 0.2},
                {"class_name": "fish", "score": 0.3},
            ],
        ]
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_multi_class_schema()
    )
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.MULTI_CLASS
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidMultiClassPredScoreValue)

    # Check dictionary of wrong type
    data_df["prediction_scores"] = pd.Series(
        [
            [{"class_name": "dog", "score": "wrong type"}],
            [{"class_name": "cat", "score": "wrong type"}],
            [{"class_name": "fish", "score": "wrong type"}],
        ]
    )
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.MULTI_CLASS
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidType)

    # Check dictionary length
    over_max_classes = []
    for i in range(MAX_NUMBER_OF_MULTI_CLASS_CLASSES + 10):
        over_max_classes.append({"class_name": f"class_{i}", "score": 0.1})
    data_df["prediction_scores"] = pd.Series(
        [
            over_max_classes,
            over_max_classes,
            over_max_classes,
        ]
    )
    data_df["threshold_scores"] = pd.Series(
        [
            over_max_classes,
            over_max_classes,
            over_max_classes,
        ]
    )
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.MULTI_CLASS
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidNumClassesMultiClassMap)
    # reset
    data_df["prediction_scores"] = pd.Series(
        [
            [
                {"class_name": "dog", "score": 0.1},
                {"class_name": "cat", "score": 0.2},
                {"class_name": "fish", "score": 0.3},
            ],
            [
                {"class_name": "dog", "score": 0.1},
                {"class_name": "cat", "score": 0.2},
                {"class_name": "fish", "score": 0.3},
            ],
            [
                {"class_name": "dog", "score": 0.1},
                {"class_name": "cat", "score": 0.2},
                {"class_name": "fish", "score": 0.3},
            ],
        ]
    )
    data_df["threshold_scores"] = data_df["prediction_scores"]
    # test invalid actual scores (must be 0 or 1)
    data_df["actual_scores"] = pd.Series(
        [
            [{"class_name": "dog", "score": 0.1}],
            [{"class_name": "fish", "score": 0.3}],
            [{"class_name": "dog", "score": 0.1}],
        ]
    )
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.MULTI_CLASS
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidMultiClassActScoreValue)

    # test empty class
    data_df["actual_scores"] = pd.Series(
        [
            [{"class_name": "dog", "score": 0}],
            [{"class_name": "", "score": 0}],
            [{"class_name": "dog", "score": 1}],
        ]
    )
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.MULTI_CLASS
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidMultiClassClassNameLength)

    # reset
    data_df["actual_scores"] = pd.Series(
        [
            [
                {"class_name": "dog", "score": 0},
                {"class_name": "cat", "score": 0},
                {"class_name": "fish", "score": 1},
            ],
            [
                {"class_name": "dog", "score": 0},
                {"class_name": "cat", "score": 1},
                {"class_name": "fish", "score": 0},
            ],
            [
                {"class_name": "dog", "score": 1},
                {"class_name": "cat", "score": 0},
                {"class_name": "fish", "score": 0},
            ],
        ]
    )
    # threshold missing class
    data_df["threshold_scores"] = pd.Series(
        [
            [
                {"class_name": "dog", "score": 0.1},
                {"class_name": "fish", "score": 0.3},
            ],
            [
                {"class_name": "dog", "score": 0.1},
                {"class_name": "fish", "score": 0.3},
            ],
            [
                {"class_name": "dog", "score": 0.1},
                {"class_name": "fish", "score": 0.3},
            ],
        ]
    )
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.MULTI_CLASS
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidMultiClassThresholdClasses)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_zero_errors_typed_schema():
    df = get_base_df()
    s = Schema(
        feature_column_names=TypedColumns(
            to_float=["A"],  # ints -> float
            to_int=["E"],  # ints with None -> int
            to_str=["C"],  # NaN floats -> NaN strings
            inferred=["F"],
        ),
        tag_column_names=["G", "H"],
    )
    schema = _overwrite_schema_fields(get_base_schema(), s)
    response_df = None
    try:
        response = log_dataframe(
            df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        pytest.fail("Unexpected error")

    # column A - column type is now nullable float64, all values are type np.float64
    assert response_df["A"].dtype == "Float64"
    assert isinstance(response_df["A"][0], np.float64)

    # column E - column type is now nullable int64, value types are np.int64 or pd.NA
    assert response_df["E"].dtype == "Int64"
    assert isinstance(response_df["E"][0], np.int64)
    assert response_df["E"][1] is pd.NA

    # column C - column type is now nullable string, value types are str or pd.NA
    assert response_df["C"].dtype == "string"
    assert response_df["C"][0] is pd.NA

    # column F - column type has not changed
    assert response_df["F"].dtype == "float64"
    assert isinstance(response_df["F"][0], np.float64)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_typed_schema_validation_empty_class():
    df = get_base_df()
    s = Schema(
        feature_column_names=TypedColumns(
            to_float=["A"],  # ints -> float
            to_int=["E"],  # ints with None -> int
            to_str=["C"],  # NaN floats -> NaN strings
        ),
        tag_column_names=TypedColumns(),
    )
    schema = _overwrite_schema_fields(get_base_schema(), s)
    with pytest.raises(InvalidTypedColumnsError) as excinfo:
        _ = log_dataframe(
            df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
    assert excinfo.value.field_name == "tag_column_names"
    assert excinfo.value.reason == "is empty"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_typed_schema_validation_empty_fields():
    df = get_base_df()
    s = Schema(
        feature_column_names=TypedColumns(
            to_int=[],
            to_str=[],
        )
    )
    schema = _overwrite_schema_fields(get_base_schema(), s)
    with pytest.raises(InvalidTypedColumnsError) as excinfo:
        _ = log_dataframe(
            df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
    assert excinfo.value.field_name == "feature_column_names"
    assert excinfo.value.reason == "is empty"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_typed_schema_validation_duplicates():
    df = get_base_df()
    s = Schema(
        feature_column_names=TypedColumns(
            to_float=["A"],
            to_str=["B", "A"],
        )
    )
    schema = _overwrite_schema_fields(get_base_schema(), s)
    with pytest.raises(InvalidTypedColumnsError) as excinfo:
        _ = log_dataframe(
            df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
    assert excinfo.value.field_name == "feature_column_names"
    assert excinfo.value.reason == "has duplicate column names: A"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_typed_schema_casting_error():
    df = get_base_df()
    g = pd.DataFrame({"g": pd.Series([0.1, 1.0, 2.0])})
    df2 = df.join(g)
    s = Schema(
        feature_column_names=TypedColumns(
            to_int=["g"],  # float -> int
        ),
        tag_column_names=["G", "H"],
    )
    schema = _overwrite_schema_fields(get_base_schema(), s)
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            df2, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
    assert isinstance(excinfo.value, ColumnCastingError)
    assert excinfo.value.attempted_casting_columns == ["g"]
    assert excinfo.value.attempted_casting_type == "Int64"
    assert (
        excinfo.value.error_msg
        == "cannot safely cast non-equivalent float64 to int64"
    )


def test_wrong_embedding_types():
    data_df = pd.concat(
        [get_base_df(), get_score_categorical_df(), get_embeddings_df()], axis=1
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    # Check embedding_vector of strings are not allowed
    data_df["image_vector"] = (
        np.random.randn(3, EMBEDDING_SIZE).astype(str).tolist()
    )
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeFeatures)
    # Reset
    data_df["image_vector"] = np.random.randn(3, EMBEDDING_SIZE).tolist()

    # Check embedding_vector of booleans are not allowed
    data_df["sentence_vector"] = np.random.choice(
        a=[True, False], size=(3, EMBEDDING_SIZE)
    ).tolist()
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeFeatures)
    # Reset
    data_df["sentence_vector"] = np.random.randn(3, EMBEDDING_SIZE).tolist()

    # Check embedding_data of float are not allowed
    data_df["sentence_data"] = [x for x in range(3)]
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeFeatures)
    # Reset
    data_df["sentence_data"] = ["data_" + str(x) for x in range(3)]

    # Check embedding_link_to_data of list of strings are not allowed
    data_df["image_link"] = [["link_"] + [str(x)] for x in range(3)]
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeFeatures)
    # Reset
    data_df["image_link"] = ["link_" + str(x) for x in range(3)]

    # Check embedding_link_to_data of float are not allowed
    data_df["image_link"] = [x for x in range(3)]
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeFeatures)
    # Reset
    data_df["image_link"] = ["link_" + str(x) for x in range(3)]

    # Check all resets were successful
    try:
        response = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        pytest.fail("Unexpected error")
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


def test_wrong_embedding_values():
    good_vector = []
    for _ in range(3):
        good_vector.append(np.arange(float(6)))

    multidimensional_vector = []
    for i in range(3):
        if i <= 1:
            multidimensional_vector.append(np.arange(float(6)))
        else:
            multidimensional_vector.append(np.arange(float(4)))

    empty_vector = []
    for _ in range(3):
        empty_vector.append(np.arange(float(0)))

    one_vector = []
    for _ in range(3):
        one_vector.append(np.arange(float(1)))

    long_vector = []
    for _ in range(3):
        long_vector.append(np.arange(float(MAX_EMBEDDING_DIMENSIONALITY + 1)))

    data_df = pd.concat([get_base_df(), get_score_categorical_df()], axis=1)
    data_df["good_vector"] = good_vector
    data_df["multidimensional_vector"] = multidimensional_vector
    data_df["empty_vector"] = empty_vector
    data_df["one_vector"] = one_vector
    data_df["long_vector"] = long_vector

    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(
        schema,
        Schema(
            embedding_feature_column_names={
                "good_embedding": EmbeddingColumnNames(
                    vector_column_name="good_vector",
                ),
                "multidimensional_embedding": EmbeddingColumnNames(
                    vector_column_name="multidimensional_vector",
                ),
                "empty_embedding": EmbeddingColumnNames(
                    vector_column_name="empty_vector",
                ),
                "one_embedding": EmbeddingColumnNames(
                    vector_column_name="one_vector",
                ),
                "long_embedding": EmbeddingColumnNames(
                    vector_column_name="long_vector",
                ),
            },
        ),
    )

    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidValueEmbeddingVectorDimensionality)
        assert "one_vector" in e.error_message()
        assert "long_vector" in e.error_message()


def test_wrong_prompt_response_types():
    data_df = pd.concat(
        [
            get_base_df(),
            get_score_categorical_df(),
            get_embeddings_df(),
            get_generative_df(),
        ],
        axis=1,
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())

    # Check prompt_vector of strings are not allowed
    data_df["prompt_vector"] = (
        np.random.randn(3, EMBEDDING_SIZE).astype(str).tolist()
    )
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["prompt_vector"] = np.random.randn(3, EMBEDDING_SIZE).tolist()

    # Check response_vector of booleans are not allowed
    data_df["response_vector"] = np.random.choice(
        a=[True, False], size=(3, EMBEDDING_SIZE)
    ).tolist()
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["response_vector"] = np.random.randn(3, EMBEDDING_SIZE).tolist()

    # Check response_data of float are not allowed
    data_df["response_data"] = [x for x in range(3)]
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["response_data"] = ["data_" + str(x) for x in range(3)]

    # Check all resets were successful
    try:
        response = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        pytest.fail("Unexpected error")

    # use json here because some row elements are lists and are not readily comparable
    response_df = response_df.drop(columns=[GENERATED_LLM_PARAMS_JSON_COL])
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


def test_wrong_prompt_template_and_llm_config_types():
    data_df = pd.concat(
        [
            get_base_df(),
            get_score_categorical_df(),
            get_embeddings_df(),
            get_generative_df(),
        ],
        axis=1,
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())

    # Check llm_model_name of floats is not allowed
    data_df["llm_model_name"] = np.random.randn(3).tolist()
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["llm_model_name"] = ["gpt-" + str(x) for x in range(3)]

    # Check prompt_template of floats is not allowed
    data_df["prompt_template"] = np.random.randn(3).tolist()
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["prompt_template"] = [
        "This is the template with version {{version}}" for _ in range(3)
    ]

    # Check prompt_template_version of floats is not allowed
    data_df["prompt_template_version"] = np.random.randn(3).tolist()
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["prompt_template_version"] = [
        "Template {x}" + str(x) for x in range(3)
    ]

    # Check llm_params of strings is not allowed
    data_df["llm_params"] = np.random.randn(3).astype(str).tolist()
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)


def test_wrong_llm_run_metadata_types():
    data_df = pd.concat(
        [
            get_base_df(),
            get_score_categorical_df(),
            get_embeddings_df(),
            get_generative_df(),
        ],
        axis=1,
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())

    # Check total_token_count of strings is not allowed
    data_df["total_token_count"] = [str(x) for x in range(3)]
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["total_token_count"] = [x * 30 for x in range(3)]

    # Check prompt_token_count of strings is not allowed
    data_df["prompt_token_count"] = [str(x) for x in range(3)]
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["prompt_token_count"] = [x * 20 for x in range(3)]

    # Check response_token_count of strings is not allowed
    data_df["response_token_count"] = [str(x) for x in range(3)]
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["response_token_count"] = [x * 10 for x in range(3)]

    # Check response_latency_ms of strings is not allowed
    data_df["response_latency_ms"] = [str(x) for x in range(3)]
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)


def test_reserved_columns_llm_run_metadata():
    data_df = pd.concat(
        [
            get_base_df(),
            get_score_categorical_df(),
            get_embeddings_df(),
            get_generative_df(),
        ],
        axis=1,
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())

    schema = _overwrite_schema_fields(
        schema,
        Schema(
            feature_column_names=[
                LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
                LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
            ],
            tag_column_names=[
                LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
                LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
            ],
        ),
    )
    data_df[LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME] = [
        str(x) for x in range(3)
    ]
    data_df[LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME] = [
        str(x) for x in range(3)
    ]
    data_df[LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME] = [
        str(x) for x in range(3)
    ]
    data_df[LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME] = [
        str(x) for x in range(3)
    ]

    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    for e in excinfo.value.errors:
        assert isinstance(e, err.ReservedColumns)
        assert len(e.reserved_columns) == 4


def test_valid_generative_prompt_response():
    data_df = pd.concat(
        [
            get_base_df(),
            get_score_categorical_df(),
            get_embeddings_df(),
            get_generative_df(),
        ],
        axis=1,
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())
    # prompt type: EmbeddingColumnNames
    # response type: EmbeddingColumnNames
    try:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    except Exception:
        pytest.fail("Unexpected error")

    # prompt type: EmbeddingColumnNames
    # response type: EmbeddingColumnNames without data
    schema = schema.replace(
        response_column_names=EmbeddingColumnNames(
            vector_column_name="response_vector",
        ),
    )
    try:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    except Exception:
        pytest.fail("Unexpected error")

    # prompt type: str
    # response type: EmbeddingColumnNames
    schema = schema.replace(
        prompt_column_names="prompt_data",
        response_column_names=EmbeddingColumnNames(
            vector_column_name="response_vector",
            data_column_name="response_data",
        ),
    )
    try:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    except Exception:
        pytest.fail("Unexpected error")

    # prompt type: EmbeddingColumnNames
    # response type: str
    schema = schema.replace(
        prompt_column_names=EmbeddingColumnNames(
            vector_column_name="prompt_vector",
            data_column_name="prompt_data",
        ),
        response_column_names="response_data",
    )
    try:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    except Exception:
        pytest.fail("Unexpected error")

    # prompt type: str
    # response type: str
    schema = schema.replace(prompt_column_names="prompt_data")
    try:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    except Exception:
        pytest.fail("Unexpected error")


def test_invalid_generative_prompt_response_types():
    data_df = pd.concat(
        [
            get_base_df(),
            get_score_categorical_df(),
            get_embeddings_df(),
            get_generative_df(),
        ],
        axis=1,
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())
    # prompt type: EmbeddingColumnNames (containing wrong types)
    # response type: EmbeddingColumnNames (containing wrong types)
    schema = schema.replace(
        prompt_column_names=EmbeddingColumnNames(
            vector_column_name="prompt_data",
            data_column_name="prompt_vector",
        ),
        response_column_names=EmbeddingColumnNames(
            vector_column_name="response_data",
            data_column_name="response_vector",
        ),
    )
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    assert len(excinfo.value.errors) == 2
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)
    # prompt type: str (containing wrong types)
    # response type: EmbeddingColumnNames (containing wrong types)
    schema = schema.replace(prompt_column_names="prompt_vector")
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    assert len(excinfo.value.errors) == 3
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)

    # prompt type: EmbeddingColumnNames (containing wrong types)
    # response type: str (containing wrong types)
    schema = schema.replace(
        prompt_column_names=EmbeddingColumnNames(
            vector_column_name="prompt_data",
            data_column_name="prompt_vector",
        ),
        response_column_names="response_vector",
    )
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    assert len(excinfo.value.errors) == 3
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)

    # prompt type: str (containing wrong types)
    # response type: str (containing wrong types)
    schema = schema.replace(prompt_column_names="prompt_vector")
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    assert len(excinfo.value.errors) == 1
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidTypeColumns)


def test_invalid_generative_prompt_response_values():
    data_df = pd.concat(
        [
            get_base_df(),
            get_score_categorical_df(),
            get_embeddings_df(),
            get_generative_df(),
        ],
        axis=1,
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())
    data_df["prompt_vector"] = np.random.randn(
        3, MAX_EMBEDDING_DIMENSIONALITY + 1
    ).tolist()
    data_df["response_vector"] = np.random.randn(
        3, MAX_EMBEDDING_DIMENSIONALITY + 1
    ).tolist()
    data_df["prompt_data"] = [
        "x" * (MAX_RAW_DATA_CHARACTERS + 1) for _ in range(3)
    ]
    data_df["response_data"] = [
        "x" * (MAX_RAW_DATA_CHARACTERS + 1) for _ in range(3)
    ]
    # prompt type: EmbeddingColumnNames (containing wrong values)
    # response type: EmbeddingColumnNames (containing wrong values)
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    assert len(excinfo.value.errors) == 2
    for e in excinfo.value.errors:
        assert isinstance(
            e,
            (
                err.InvalidValueEmbeddingVectorDimensionality,
                err.InvalidValueEmbeddingRawDataTooLong,
            ),
        )
    # prompt type: str (containing wrong values)
    # response type: EmbeddingColumnNames (containing wrong values)
    schema = schema.replace(prompt_column_names="prompt_data")
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    assert len(excinfo.value.errors) == 2
    for e in excinfo.value.errors:
        assert isinstance(
            e,
            (
                err.InvalidValueEmbeddingVectorDimensionality,
                err.InvalidValueEmbeddingRawDataTooLong,
            ),
        )

    # prompt type: EmbeddingColumnNames (containing wrong values)
    # response type: str (containing wrong values)
    schema = schema.replace(
        prompt_column_names=EmbeddingColumnNames(
            vector_column_name="prompt_vector",
            data_column_name="prompt_data",
        ),
        response_column_names="response_data",
    )
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    assert len(excinfo.value.errors) == 2
    for e in excinfo.value.errors:
        assert isinstance(
            e,
            (
                err.InvalidValueEmbeddingVectorDimensionality,
                err.InvalidValueEmbeddingRawDataTooLong,
            ),
        )

    # prompt type: str (containing wrong values)
    # response type: str (containing wrong values)
    schema = schema.replace(prompt_column_names="prompt_data")
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    assert len(excinfo.value.errors) == 1
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidValueEmbeddingRawDataTooLong)


def test_generative_without_prompt_response():
    data_df = pd.concat(
        [
            get_base_df(),
            get_score_categorical_df(),
            get_embeddings_df(),
            get_generative_df(),
        ],
        axis=1,
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())
    schema = schema.replace(prompt_column_names=None)
    schema = schema.replace(response_column_names=None)

    try:
        response = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        pytest.fail("Unexpected error")
    response_df = response_df.drop(
        columns=[
            GENERATED_LLM_PARAMS_JSON_COL,
        ]
    )
    data_df = data_df.drop(
        columns=[
            "prompt_vector",
            "prompt_data",
            "response_vector",
            "response_data",
        ]
    )
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


def test_generative_default_prediction_label():
    @dataclass
    class TestConfig:
        name: str
        should_have_generated_default_prediction_label: bool
        drop_columns: List[str]
        replace_column_names: Dict
        # generated columns that need to be dropped in order to perform df equality check between
        # input data and response data
        drop_generated_columns: List[str]

    tests = [
        TestConfig(
            name="no_prediction_label_or_actual_label",
            should_have_generated_default_prediction_label=True,
            drop_columns=["prediction_label", "actual_label"],
            replace_column_names={
                "prediction_label_column_name": None,
                "actual_label_column_name": None,
            },
            drop_generated_columns=[
                GENERATED_LLM_PARAMS_JSON_COL,
                GENERATED_PREDICTION_LABEL_COL,
            ],
        ),
        # Testing the case that allows us to still support latent actual
        TestConfig(
            name="no_prediction_label",
            should_have_generated_default_prediction_label=False,
            drop_columns=["prediction_label"],
            replace_column_names={"prediction_label_column_name": None},
            drop_generated_columns=[GENERATED_LLM_PARAMS_JSON_COL],
        ),
        TestConfig(
            name="no_actual_label",
            should_have_generated_default_prediction_label=False,
            drop_columns=["actual_label"],
            replace_column_names={"actual_label_column_name": None},
            drop_generated_columns=[GENERATED_LLM_PARAMS_JSON_COL],
        ),
        TestConfig(
            name="include_prediction_and_actual",
            should_have_generated_default_prediction_label=False,
            drop_columns=[],
            replace_column_names={},
            drop_generated_columns=[GENERATED_LLM_PARAMS_JSON_COL],
        ),
    ]

    for test in tests:
        data_df = pd.concat(
            [
                get_base_df(),
                get_score_categorical_df(),
                get_embeddings_df(),
                get_generative_df(),
            ],
            axis=1,
        ).drop(columns=test.drop_columns)
        schema = _overwrite_schema_fields(
            get_base_schema(), get_score_categorical_schema()
        )
        schema = _overwrite_schema_fields(schema, get_embeddings_schema())
        schema = _overwrite_schema_fields(schema, get_generative_schema())
        # Check logging without prediction and actual label; validate that a default prediction label
        # is added.
        schema = schema.replace(**test.replace_column_names)

        try:
            response = log_dataframe(
                data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
            )
            response_df: pd.DataFrame = response.df  # type:ignore
        except Exception:
            pytest.fail("Unexpected error")
        if test.should_have_generated_default_prediction_label:
            # Check if default_prediction_labels are generated
            if GENERATED_PREDICTION_LABEL_COL not in response_df.columns:
                pytest.fail("Unexpected error")
            # Check all values of default_prediction_label are equal to 1
            if (response_df[GENERATED_PREDICTION_LABEL_COL] != 1).any():
                pytest.fail("Unexpected error")

        response_df = response_df.drop(columns=test.drop_generated_columns)
        assert (
            response_df.sort_index(axis=1).to_json()
            == roundtrip_df(data_df).sort_index(axis=1).to_json()
        ), test.name


def test_invalid_generative_llm_prompt_template_and_llm_config_values():
    data_df = pd.concat(
        [
            get_base_df(),
            get_score_categorical_df(),
            get_embeddings_df(),
            get_generative_df(),
        ],
        axis=1,
    )
    schema = _overwrite_schema_fields(
        get_base_schema(), get_score_categorical_schema()
    )
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())

    # Check fields with too many characters is not allowed
    data_df["llm_model_name"] = pd.Series(
        ["a" * (MAX_LLM_MODEL_NAME_LENGTH + 1)] * 3
    )
    data_df["prompt_template"] = pd.Series(
        ["a" * (MAX_PROMPT_TEMPLATE_LENGTH + 1)] * 3
    )
    data_df["prompt_template_version"] = pd.Series(
        ["a" * (MAX_PROMPT_TEMPLATE_VERSION_LENGTH + 1)] * 3
    )
    with pytest.raises(Exception) as excinfo:
        _ = log_dataframe(
            data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM
        )
    assert isinstance(excinfo.value, err.ValidationFailure)
    assert len(excinfo.value.errors) == 3
    for e in excinfo.value.errors:
        assert isinstance(e, err.InvalidStringLengthInColumn)


def get_stubbed_client(additional_headers=None):
    c = Client(
        space_key="space_key",
        api_key="api_key",
        additional_headers=additional_headers,
    )

    def _post_file(path, schema):
        return 200

    c._post_file = _post_file
    return c


def test_instantiating_client_duplicated_header():
    with pytest.raises(Exception) as excinfo:
        _ = get_stubbed_client({"authorization": "FAKE_VALUE"})
    assert (
        "Found invalid additional header, cannot use reserved headers named: authorization."
        in str(excinfo.value)
    )


def test_instantiating_client_additional_header():
    c = get_stubbed_client({"JWT": "FAKE_VALUE"})
    expected = {
        "authorization": "api_key",
        "arize-space-id": None,
        "space_id": None,
        "arize-space-key": "space_key",
        "space": "space_key",
        "sdk-language": "python",
        "language-version": get_python_version(),
        "sdk-version": arize_version,
        "sync": "0",  # Defaults to async logging
        "JWT": "FAKE_VALUE",
        "arize-interface": "batch",
    }
    assert c._headers == expected


def test_invalid_client_auth_log_method_passed_vars():
    with pytest.raises(AuthError) as excinfo:
        c = Client()
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log(
            dataframe="dummy",
            schema="dummy",
            environment="dummy",
            model_id="dummy",
            model_type="dummy",
        )
    # if all missing - prompt for api_key and space_id
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_api_key=True,
            missing_space_id=True,
            method_name="log",
        ).error_message()
    )

    with pytest.raises(AuthError) as excinfo:
        c = Client(space_id="space_id")
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log(
            dataframe="dummy",
            schema="dummy",
            environment="dummy",
            model_id="dummy",
            model_type="dummy",
        )
    # if all missing - prompt for api_key and space_id
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_api_key=True,
            method_name="log",
        ).error_message()
    )
    with pytest.raises(AuthError) as excinfo:
        c = Client(space_key="space_key")
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log(
            dataframe="dummy",
            schema="dummy",
            environment="dummy",
            model_id="dummy",
            model_type="dummy",
        )
    # if all missing - prompt for api_key and space_id
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_api_key=True,
            method_name="log",
        ).error_message()
    )

    # if both space_key and space_id are missing, promt only for space_id
    with pytest.raises(AuthError) as excinfo:
        c = Client(api_key="api_key")
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log(
            dataframe="dummy",
            schema="dummy",
            environment="dummy",
            model_id="dummy",
            model_type="dummy",
        )
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_space_id=True,
            method_name="log",
        ).error_message()
    )

    # acceptable input
    try:
        _ = Client(space_key="space_key", api_key="api_key")
    except Exception:
        pytest.fail("Unexpected error!")


def test_invalid_client_auth_log_spans_method_passed_vars():
    with pytest.raises(AuthError) as excinfo:
        c = Client()
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log_spans(
            dataframe="dummy",
        )
    # if all missing - prompt for api_key and space_id
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_api_key=True,
            missing_space_id=True,
            method_name="log_spans",
        ).error_message()
    )

    with pytest.raises(AuthError) as excinfo:
        c = Client(space_id="space_id")
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log_spans(
            dataframe="dummy",
        )
    # if all missing - prompt for api_key and space_id
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_api_key=True,
            method_name="log_spans",
        ).error_message()
    )
    with pytest.raises(AuthError) as excinfo:
        c = Client(space_key="space_key")
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log_spans(
            dataframe="dummy",
        )
    # if all missing - prompt for api_key and space_id
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_api_key=True,
            method_name="log_spans",
        ).error_message()
    )

    # if both space_key and space_id are missing, promt only for space_id
    with pytest.raises(AuthError) as excinfo:
        c = Client(api_key="api_key")
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log_spans(
            dataframe="dummy",
        )
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_space_id=True,
            method_name="log_spans",
        ).error_message()
    )

    # acceptable input
    try:
        _ = Client(space_key="space_key", api_key="api_key")
    except Exception:
        pytest.fail("Unexpected error!")


def test_invalid_client_auth_log_evaluations_method_passed_vars():
    with pytest.raises(AuthError) as excinfo:
        c = Client()
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log_evaluations(
            dataframe="dummy",
        )
    # if all missing - prompt for api_key and space_id
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_api_key=True,
            missing_space_id=True,
            method_name="log_evaluations",
        ).error_message()
    )

    with pytest.raises(AuthError) as excinfo:
        c = Client(space_id="space_id")
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log_evaluations(
            dataframe="dummy",
        )
    # if all missing - prompt for api_key and space_id
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_api_key=True,
            method_name="log_evaluations",
        ).error_message()
    )
    with pytest.raises(AuthError) as excinfo:
        c = Client(space_key="space_key")
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log_evaluations(
            dataframe="dummy",
        )
    # if all missing - prompt for api_key and space_id
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_api_key=True,
            method_name="log_evaluations",
        ).error_message()
    )

    # if both space_key and space_id are missing, promt only for space_id
    with pytest.raises(AuthError) as excinfo:
        c = Client(api_key="api_key")
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log_evaluations(
            dataframe="dummy",
        )
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_space_id=True,
            method_name="log_evaluations",
        ).error_message()
    )

    # acceptable input
    try:
        _ = Client(space_key="space_key", api_key="api_key")
    except Exception:
        pytest.fail("Unexpected error!")


def test_invalid_client_auth_log_evaluations_sync_method_passed_vars():
    with pytest.raises(AuthError) as excinfo:
        c = Client()
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log_evaluations_sync(
            dataframe="dummy",
        )
    # if all missing - prompt for api_key and space_id
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_space_id=True,
            missing_developer_key=True,
            method_name="log_evaluations_sync",
        ).error_message()
    )

    with pytest.raises(AuthError) as excinfo:
        c = Client(space_id="space_id")
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log_evaluations_sync(
            dataframe="dummy",
        )
    # if all missing - prompt for api_key and space_id
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_developer_key=True,
            method_name="log_evaluations_sync",
        ).error_message()
    )
    with pytest.raises(AuthError) as excinfo:
        c = Client(space_key="space_key")
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log_evaluations_sync(
            dataframe="dummy",
        )
    # if all missing - prompt for api_key and space_id
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_space_id=True,
            missing_developer_key=True,
            method_name="log_evaluations_sync",
        ).error_message()
    )

    # if both space_key and space_id are missing, promt only for space_id
    with pytest.raises(AuthError) as excinfo:
        c = Client(api_key="api_key")
        # The first action is to validate the keys, the inputs don't matter
        _ = c.log_evaluations_sync(
            dataframe="dummy",
        )
    assert (
        excinfo.value.__str__()
        == AuthError(
            missing_space_id=True,
            missing_developer_key=True,
            method_name="log_evaluations_sync",
        ).error_message()
    )

    # acceptable input
    try:
        _ = Client(
            space_id="space_id",
            developer_key="developer_key",
            host="host",
            port=1234,  # not a real port
        )
    except Exception:
        pytest.fail("Unexpected error!")


def _overwrite_schema_fields(schema1: Schema, schema2: Schema) -> Schema:
    """This function overwrites a base Schema `schema1` with the fields of `schema2`
    that are not None

    Arguments:
    ----------
        schema1 (Schema): Base Schema with fields to be overwritten
        schema2 (Schema): New Schema used to overwrite schema1

    Returns:
    --------
        Schema: The resulting schema
    """
    schema_dict_fields = (
        "embedding_feature_column_names",
        "shap_values_column_names",
        "prompt_column_names",
        "response_column_names",
        "prompt_template_column_names",
        "llm_config_column_names",
        "llm_run_metadata_column_names",
    )
    changes = {
        k: v
        for k, v in schema2.asdict().items()
        if v is not None and k not in schema_dict_fields
    }
    schema = schema1.replace(**changes)

    # Embedding column names need to be treated separately for being in a dictionary
    if schema2.embedding_feature_column_names is not None:
        emb_feat_col_names = schema1.embedding_feature_column_names
        if emb_feat_col_names is None:
            emb_feat_col_names = {}
        for k, v in schema2.embedding_feature_column_names.items():
            emb_feat_col_names[k] = v
        # replace embedding column names in schema
        schema = schema.replace(
            embedding_feature_column_names=emb_feat_col_names
        )

    # Prompt and response need to be treated separately
    if schema2.prompt_column_names is not None:
        schema = schema.replace(prompt_column_names=schema2.prompt_column_names)
    if schema2.response_column_names is not None:
        schema = schema.replace(
            response_column_names=schema2.response_column_names
        )

    # Shap values column names need to be treated separately for being in a dictionary
    if schema2.shap_values_column_names is not None:
        shap_val_col_names = schema1.shap_values_column_names
        if shap_val_col_names is None:
            shap_val_col_names = {}
        for k, v in schema2.shap_values_column_names.items():
            shap_val_col_names[k] = v
        # replace embedding column names in schema
        schema = schema.replace(shap_values_column_names=shap_val_col_names)

    # prompt_template_column_names, llm_config_column_names, and llm_run_metadata_column_names
    # need to be treated separately
    if schema2.prompt_template_column_names is not None:
        schema = schema.replace(
            prompt_template_column_names=schema2.prompt_template_column_names
        )
    if schema2.llm_config_column_names is not None:
        schema = schema.replace(
            llm_config_column_names=schema2.llm_config_column_names
        )
    if schema2.llm_run_metadata_column_names is not None:
        schema = schema.replace(
            llm_run_metadata_column_names=schema2.llm_run_metadata_column_names
        )

    # feature_column_names and tag_column_names need to be treated separately if they are of type TypedColumns
    if schema2.feature_column_names is not None and isinstance(
        schema2.feature_column_names, TypedColumns
    ):
        schema = schema.replace(
            feature_column_names=schema2.feature_column_names
        )
    if schema2.tag_column_names is not None and isinstance(
        schema2.tag_column_names, TypedColumns
    ):
        schema = schema.replace(tag_column_names=schema2.tag_column_names)

    return schema


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
