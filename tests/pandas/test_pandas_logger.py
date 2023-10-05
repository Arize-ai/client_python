import datetime
import sys
import uuid

import arize.pandas.validation.errors as err
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from arize import __version__ as arize_version
from arize.pandas.logger import Client
from arize.utils.constants import (
    API_KEY_ENVVAR_NAME,
    GENERATED_LLM_PARAMS_JSON_COL,
    GENERATED_PREDICTION_LABEL_COL,
    MAX_LLM_MODEL_NAME_LENGTH,
    MAX_PROMPT_TEMPLATE_LENGTH,
    MAX_PROMPT_TEMPLATE_VERSION_LENGTH,
    SPACE_KEY_ENVVAR_NAME,
)
from arize.utils.errors import AuthError
from arize.utils.types import (
    EmbeddingColumnNames,
    Environments,
    LLMConfigColumnNames,
    ModelTypes,
    ObjectDetectionColumnNames,
    PromptTemplateColumnNames,
    Schema,
)
from arize.utils.utils import get_python_version
from requests import Response

EMBEDDING_SIZE = 15


class MockResponse(Response):
    def __init__(self, df, reason, status_code):
        super().__init__()
        self.df = df
        self.reason = reason
        self.status_code = status_code


class NoSendClient(Client):
    def _post_file(self, path, schema, sync, timeout):
        return MockResponse(pa.ipc.open_stream(pa.OSFile(path)).read_pandas(), "Success", 200)


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
            "I": pd.Categorical(["a", "b", "c"], ordered=True, categories=["c", "b", "a"]),
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
            "sentence_vector": [np.random.randn(EMBEDDING_SIZE) for x in range(3)],
            "sentence_data": ["data_" + str(x) for x in range(3)],
            "token_array_vector": [np.random.randn(EMBEDDING_SIZE) for x in range(3)],
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


def get_generative_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt_vector": np.random.randn(3, EMBEDDING_SIZE).tolist(),
            "response_vector": np.random.randn(3, EMBEDDING_SIZE).tolist(),
            "prompt_data": ["data_" + str(x) for x in range(3)],
            "response_data": ["data_" + str(x) for x in range(3)],
            "llm_model_name": ["gpt-" + str(x) for x in range(3)],
            "prompt_template": ["This is the template with version {{version}}" for _ in range(3)],
            "prompt_template_version": ["Template {x}" + str(x) for x in range(3)],
            "llm_params": [
                {
                    "presence_penalty": x / 3,
                    "stop": [".", "?", "!"],
                    "temperature": x / 4,
                }
                for x in range(3)
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
    )


# roundtrip df is the expected df that would be re-constructed from the pyarrow serialization, where
# 1. the column excluded from schema should have been dropped
# 2. categarical variables should have been converted to string
def roundtrip_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("excluded_from_schema", axis=1, errors="ignore")
    return df.astype({k: "str" for k, v in df.dtypes.items() if v.name == "category"})


def log_dataframe(
    df: pd.DataFrame, schema: Schema, model_type: ModelTypes, environment=Environments.PRODUCTION
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
    data_df = pd.concat([get_base_df(), get_score_categorical_df(), get_embeddings_df()], axis=1)
    schema = _overwrite_schema_fields(get_base_schema(), get_score_categorical_schema())
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    try:
        response = log_dataframe(data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL)
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        assert False
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )

    # CHECK logging without prediction_id
    schema = schema.replace(prediction_id_column_name=None)
    try:
        response = log_dataframe(data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL)
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        assert False
    data_df = data_df.drop(columns=["prediction_id"])
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )

    # CHECK logging generative model
    data_df = pd.concat(
        [get_base_df(), get_score_categorical_df(), get_embeddings_df(), get_generative_df()],
        axis=1,
    )
    schema = _overwrite_schema_fields(get_base_schema(), get_score_categorical_schema())
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())
    try:
        response = log_dataframe(data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM)
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        assert False
    # use json here because some row elements are lists and are not readily comparable
    response_df = response_df.drop(columns=[GENERATED_LLM_PARAMS_JSON_COL])
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_zero_errors_object_detection():
    # CHECK logging object detection model
    data_df = pd.concat([get_base_df(), get_object_detection_df()], axis=1)
    schema = _overwrite_schema_fields(get_base_schema(), get_object_detection_schema())
    try:
        response = log_dataframe(data_df, schema=schema, model_type=ModelTypes.OBJECT_DETECTION)
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        assert False
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


def test_wrong_embedding_types():
    data_df = pd.concat([get_base_df(), get_score_categorical_df(), get_embeddings_df()], axis=1)
    schema = _overwrite_schema_fields(get_base_schema(), get_score_categorical_schema())
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    # Check embedding_vector of strings are not allowed
    data_df["image_vector"] = np.random.randn(3, EMBEDDING_SIZE).astype(str).tolist()
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidTypeFeatures)
    # Reset
    data_df["image_vector"] = np.random.randn(3, EMBEDDING_SIZE).tolist()

    # Check embedding_vector of booleans are not allowed
    data_df["sentence_vector"] = np.random.choice(
        a=[True, False], size=(3, EMBEDDING_SIZE)
    ).tolist()
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidTypeFeatures)
    # Reset
    data_df["sentence_vector"] = np.random.randn(3, EMBEDDING_SIZE).tolist()

    # Check embedding_data of float are not allowed
    data_df["sentence_data"] = [x for x in range(3)]
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidTypeFeatures)
    # Reset
    data_df["sentence_data"] = ["data_" + str(x) for x in range(3)]

    # Check embedding_link_to_data of list of strings are not allowed
    data_df["image_link"] = [["link_"] + [str(x)] for x in range(3)]
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidTypeFeatures)
    # Reset
    data_df["image_link"] = ["link_" + str(x) for x in range(3)]

    # Check embedding_link_to_data of float are not allowed
    data_df["image_link"] = [x for x in range(3)]
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidTypeFeatures)
    # Reset
    data_df["image_link"] = ["link_" + str(x) for x in range(3)]

    # Check all resets were successful
    try:
        response = log_dataframe(data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL)
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        assert False
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


def test_wrong_embedding_values():
    good_vector = []
    for i in range(3):
        good_vector.append(np.arange(float(6)))

    multidimensional_vector = []
    for i in range(3):
        if i <= 1:
            multidimensional_vector.append(np.arange(float(6)))
        else:
            multidimensional_vector.append(np.arange(float(4)))

    empty_vector = []
    for i in range(3):
        empty_vector.append(np.arange(float(0)))

    one_vector = []
    for i in range(3):
        one_vector.append(np.arange(float(1)))

    data_df = pd.concat([get_base_df(), get_score_categorical_df()], axis=1)
    data_df["good_vector"] = good_vector
    data_df["multidimensional_vector"] = multidimensional_vector
    data_df["empty_vector"] = empty_vector
    data_df["one_vector"] = one_vector

    schema = _overwrite_schema_fields(get_base_schema(), get_score_categorical_schema())
    schema = _overwrite_schema_fields(
        schema,
        Schema(
            embedding_feature_column_names={
                "good_embedding": EmbeddingColumnNames(
                    vector_column_name="good_vector",
                ),
                "multidimensional_embedding": EmbeddingColumnNames(
                    vector_column_name="multidimensional_vector",  # Should give error
                ),
                "empty_embedding": EmbeddingColumnNames(
                    vector_column_name="empty_vector",  # Should give error
                ),
                "one_embedding": EmbeddingColumnNames(
                    vector_column_name="one_vector",  # Should give error
                ),
            },
        ),
    )

    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.SCORE_CATEGORICAL)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidValueLowEmbeddingVectorDimensionality)


def test_wrong_prompt_response_types():
    data_df = pd.concat(
        [get_base_df(), get_score_categorical_df(), get_embeddings_df(), get_generative_df()],
        axis=1,
    )
    schema = _overwrite_schema_fields(get_base_schema(), get_score_categorical_schema())
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())

    # Check prompt_vector of strings are not allowed
    data_df["prompt_vector"] = np.random.randn(3, EMBEDDING_SIZE).astype(str).tolist()
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["prompt_vector"] = np.random.randn(3, EMBEDDING_SIZE).tolist()

    # Check response_vector of booleans are not allowed
    data_df["response_vector"] = np.random.choice(
        a=[True, False], size=(3, EMBEDDING_SIZE)
    ).tolist()
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["response_vector"] = np.random.randn(3, EMBEDDING_SIZE).tolist()

    # Check response_data of float are not allowed
    data_df["response_data"] = [x for x in range(3)]
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["response_data"] = ["data_" + str(x) for x in range(3)]

    # Check all resets were successful
    try:
        response = log_dataframe(data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM)
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        assert False

    # use json here because some row elements are lists and are not readily comparable
    response_df = response_df.drop(columns=[GENERATED_LLM_PARAMS_JSON_COL])
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


def test_wrong_prompt_template_and_llm_config_types():
    data_df = pd.concat(
        [get_base_df(), get_score_categorical_df(), get_embeddings_df(), get_generative_df()],
        axis=1,
    )
    schema = _overwrite_schema_fields(get_base_schema(), get_score_categorical_schema())
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())

    # Check llm_model_name of floats is not allowed
    data_df["llm_model_name"] = np.random.randn(3).tolist()
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["llm_model_name"] = ["gpt-" + str(x) for x in range(3)]

    # Check prompt_template of floats is not allowed
    data_df["prompt_template"] = np.random.randn(3).tolist()
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["prompt_template"] = ["This is the template with version {{version}}" for _ in range(3)]

    # Check prompt_template_version of floats is not allowed
    data_df["prompt_template_version"] = np.random.randn(3).tolist()
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidTypeColumns)
    # Reset
    data_df["prompt_template_version"] = ["Template {x}" + str(x) for x in range(3)]

    # Check llm_params of strings is not allowed
    data_df["llm_params"] = np.random.randn(3).astype(str).tolist()
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.InvalidTypeColumns)


def test_generative_without_prompt_response():
    data_df = pd.concat(
        [get_base_df(), get_score_categorical_df(), get_embeddings_df(), get_generative_df()],
        axis=1,
    )
    schema = _overwrite_schema_fields(get_base_schema(), get_score_categorical_schema())
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())
    schema = schema.replace(prompt_column_names=None)
    schema = schema.replace(response_column_names=None)

    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        for e in excpt.errors:
            assert isinstance(e, err.MissingPromptResponseGenerativeLLM)


def test_generative_without_prediction_label_column_name():
    data_df = pd.concat(
        [get_base_df(), get_score_categorical_df(), get_embeddings_df(), get_generative_df()],
        axis=1,
    ).drop(columns=["prediction_label"])
    schema = _overwrite_schema_fields(get_base_schema(), get_score_categorical_schema())
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())
    # Check logging without prediction_id
    schema = schema.replace(prediction_label_column_name=None)

    try:
        response = log_dataframe(data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM)
        response_df: pd.DataFrame = response.df  # type:ignore
    except Exception:
        assert False
    # Check if defualt_prediction_labels are generated
    if GENERATED_PREDICTION_LABEL_COL not in response_df.columns:
        assert False
    # Check all values of defualt_prediction_label are equal to 1
    if (response_df[GENERATED_PREDICTION_LABEL_COL] != 1).any():
        assert False

    response_df = response_df.drop(
        columns=[GENERATED_PREDICTION_LABEL_COL, GENERATED_LLM_PARAMS_JSON_COL]
    )
    assert (
        response_df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


def test_invalid_generative_llm_prompt_template_and_llm_config_values():
    data_df = pd.concat(
        [get_base_df(), get_score_categorical_df(), get_embeddings_df(), get_generative_df()],
        axis=1,
    )
    schema = _overwrite_schema_fields(get_base_schema(), get_score_categorical_schema())
    schema = _overwrite_schema_fields(schema, get_embeddings_schema())
    schema = _overwrite_schema_fields(schema, get_generative_schema())

    # Check fields with too many characters is not allowed
    data_df["llm_model_name"] = pd.Series(["a" * (MAX_LLM_MODEL_NAME_LENGTH + 1)] * 3)
    data_df["prompt_template"] = pd.Series(["a" * (MAX_PROMPT_TEMPLATE_LENGTH + 1)] * 3)
    data_df["prompt_template_version"] = pd.Series(
        ["a" * (MAX_PROMPT_TEMPLATE_VERSION_LENGTH + 1)] * 3
    )
    try:
        _ = log_dataframe(data_df, schema=schema, model_type=ModelTypes.GENERATIVE_LLM)
    except Exception as excpt:
        assert isinstance(excpt, err.ValidationFailure)
        assert len(excpt.errors) == 3
        for e in excpt.errors:
            assert isinstance(e, err.InvalidStringLengthInColumn)


def get_stubbed_client(additional_headers=None):
    c = Client(space_key="space_key", api_key="api_key", additional_headers=additional_headers)

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
        "space": "space_key",
        "sdk-language": "python",
        "language-version": get_python_version(),
        "sdk-version": arize_version,
        "sync": "0",  # Defaults to async logging
        "JWT": "FAKE_VALUE",
    }
    assert c._headers == expected


def test_invalid_client_auth_passed_vars():
    with pytest.raises(AuthError) as excinfo:
        _ = Client()
    assert excinfo.value.__str__() == AuthError(None, None).error_message()
    assert "Missing: ['api_key', 'space_key']" in str(excinfo.value)

    with pytest.raises(AuthError) as excinfo:
        _ = Client(space_key="space_key")
    assert excinfo.value.__str__() == AuthError(None, "space_key").error_message()
    assert "Missing: ['api_key']" in str(excinfo.value)

    with pytest.raises(AuthError) as excinfo:
        _ = Client(api_key="api_key")
    assert excinfo.value.__str__() == AuthError("api_key", None).error_message()
    assert "Missing: ['space_key']" in str(excinfo.value)

    # acceptable input
    try:
        _ = Client(space_key="space_key", api_key="api_key")
    except Exception:
        pytest.fail("Unexpected error!")


def test_invalid_client_auth_environment_vars(monkeypatch):
    with pytest.raises(AuthError) as excinfo:
        _ = Client()
    assert excinfo.value.__str__() == AuthError(None, None).error_message()
    assert "Missing: ['api_key', 'space_key']" in str(excinfo.value)

    monkeypatch.setenv(SPACE_KEY_ENVVAR_NAME, "space_key")
    with pytest.raises(AuthError) as excinfo:
        c = Client()
        assert c._space_key == "space_key"
    assert excinfo.value.__str__() == AuthError(None, "space_key").error_message()
    assert "Missing: ['api_key']" in str(excinfo.value)

    monkeypatch.delenv(SPACE_KEY_ENVVAR_NAME)
    monkeypatch.setenv(API_KEY_ENVVAR_NAME, "api_key")
    with pytest.raises(AuthError) as excinfo:
        c = Client()
        assert c._api_key == "api_key"
    assert excinfo.value.__str__() == AuthError("api_key", None).error_message()
    assert "Missing: ['space_key']" in str(excinfo.value)

    # acceptable input
    monkeypatch.setenv(SPACE_KEY_ENVVAR_NAME, "space_key")
    try:
        c = Client()
    except Exception:
        pytest.fail("Unexpected error!")
    assert c._space_key == "space_key"
    assert c._api_key == "api_key"


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
    )
    changes = {
        k: v for k, v in schema2.asdict().items() if v is not None and k not in schema_dict_fields
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
        schema = schema.replace(embedding_feature_column_names=emb_feat_col_names)

    # Prompt and response need to be treated separately
    if schema2.prompt_column_names is not None:
        schema = schema.replace(prompt_column_names=schema2.prompt_column_names)
    if schema2.response_column_names is not None:
        schema = schema.replace(response_column_names=schema2.response_column_names)

    # Shap values column names need to be treated separately for being in a dictionary
    if schema2.shap_values_column_names is not None:
        shap_val_col_names = schema1.shap_values_column_names
        if shap_val_col_names is None:
            shap_val_col_names = {}
        for k, v in schema2.shap_values_column_names.items():
            shap_val_col_names[k] = v
        # replace embedding column names in schema
        schema = schema.replace(shap_values_column_names=shap_val_col_names)

    # prompt_template_column_names and llm_config_column_names need to be treated separately
    if schema2.prompt_template_column_names is not None:
        schema = schema.replace(prompt_template_column_names=schema2.prompt_template_column_names)
    if schema2.llm_config_column_names is not None:
        schema = schema.replace(llm_config_column_names=schema2.llm_config_column_names)

    return schema


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
