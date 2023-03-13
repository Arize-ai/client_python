import datetime

import arize.pandas.validation.errors as err
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from arize.pandas.logger import Client
from arize.utils.types import (
    EmbeddingColumnNames,
    Environments,
    ModelTypes,
    ObjectDetectionColumnNames,
    Schema,
)
from requests import Response


class MockResponse(Response):
    def __init__(self, df, reason, status_code):
        super().__init__()
        self.df = df
        self.reason = reason
        self.status_code = status_code


class NoSendClient(Client):
    def _post_file(self, path, schema, sync, timeout):
        return MockResponse(pa.ipc.open_stream(pa.OSFile(path)).read_pandas(), "Success", 200)


EMBEDDING_SIZE = 15
data_df = pd.DataFrame(
    {
        "prediction_id": pd.Series([0, 1, 2]),
        "prediction_timestamp": pd.Series(
            [
                datetime.datetime.now(),
                datetime.datetime.now() - datetime.timedelta(days=364),
                datetime.datetime.now() + datetime.timedelta(days=364),
            ]
        ),
        "prediction_label": pd.Series(["fraud", "not fraud", "fraud"]),
        "prediction_score": pd.Series([0.2, 0.3, 0.4]),
        "actual_label": pd.Categorical(
            ["not fraud", "fraud", "not fraud"],
            ordered=True,
            categories=["fraud", "not fraud"],
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
        "I": pd.Categorical(["a", "b", "c"], ordered=True, categories=["c", "b", "a"]),
        #####
        "image_vector": np.random.randn(3, EMBEDDING_SIZE).tolist(),
        "image_link": ["link_" + str(x) for x in range(3)],
        "sentence_vector": [np.random.randn(EMBEDDING_SIZE) for x in range(3)],
        "sentence_data": ["data_" + str(x) for x in range(3)],
        "token_array_vector": [np.random.randn(EMBEDDING_SIZE) for x in range(3)],
        "token_array_data": [["Token", "array", str(x)] for x in range(3)],
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
data_df_object_detection = pd.DataFrame(
    {
        "prediction_id": pd.Series([0, 1, 2]),
        "prediction_timestamp": pd.Series(
            [
                datetime.datetime.now(),
                datetime.datetime.now() - datetime.timedelta(days=364),
                datetime.datetime.now() + datetime.timedelta(days=364),
            ]
        ),
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


# roundtrip df is the expected df that would be re-constructed from the pyarrow serialization, where
# 1. the column excluded from schema should have been dropped
# 2. categarical variables should have been converted to string
def roundtrip_df(df):
    df = df.drop("excluded_from_schema", axis=1, errors="ignore")
    return df.astype({k: "str" for k, v in df.dtypes.items() if v.name == "category"})


def log_dataframe(df, schema=None):
    client = NoSendClient("apikey", "spaceKey")
    if schema is None:
        schema = Schema(
            prediction_id_column_name="prediction_id",
            timestamp_column_name="prediction_timestamp",
            feature_column_names=list("ABCDEFGHI"),
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
            tag_column_names=list("ABCDEFGHI"),
            prediction_label_column_name="prediction_label",
            actual_label_column_name="actual_label",
            prediction_score_column_name="prediction_score",
            actual_score_column_name="actual_score",
            shap_values_column_names=dict(zip("ABCDEF", "abcdef")),
        )

    response = client.log(
        dataframe=df,
        model_id="model-id",
        model_version="1.0",
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=Environments.PRODUCTION,
        schema=schema,
    )
    return response


def log_object_detection_dataframe(df, schema=None):
    client = NoSendClient("apikey", "spaceKey")

    if schema is None:
        schema = Schema(
            prediction_id_column_name="prediction_id",
            timestamp_column_name="prediction_timestamp",
            feature_column_names=list("ABCDEFGHI"),
            object_detection_prediction_column_names=ObjectDetectionColumnNames(
                boxes_coords_column_name="bounding_boxes_coordinates",
                categories_column_name="bounding_boxes_categories",
                scores_column_name="bounding_boxes_scores",
            ),
            object_detection_actual_column_names=ObjectDetectionColumnNames(
                boxes_coords_column_name="bounding_boxes_coordinates",
                categories_column_name="bounding_boxes_categories",
            ),
            embedding_feature_column_names={
                # Dictionary keys will be the displayed name of the embedding feature in the app
                "image_embedding": EmbeddingColumnNames(
                    vector_column_name="image_vector",
                    link_to_data_column_name="image_link",
                ),
            },
            tag_column_names=list("ABCDEFGHI"),
        )

    response = client.log(
        dataframe=df,
        model_id="model-id",
        model_version="1.0",
        model_type=ModelTypes.OBJECT_DETECTION,
        environment=Environments.PRODUCTION,
        schema=schema,
    )
    return response


def test_production_zero_errors():
    try:
        response = log_dataframe(data_df)
    except err.ValidationFailure:
        assert False
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response.df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )

    try:
        response = log_object_detection_dataframe(data_df_object_detection)
    except err.ValidationFailure:
        assert False
    # use json here because some row elements are lists and are not readily comparable
    assert (
        response.df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df_object_detection).sort_index(axis=1).to_json()
    )


def test_production_wrong_embedding_types():
    # Check embedding_vector of strings are not allowed
    data_df["image_vector"] = np.random.randn(3, EMBEDDING_SIZE).astype(str).tolist()
    try:
        _ = log_dataframe(data_df)
    except Exception as e:
        assert isinstance(e, err.ValidationFailure)
    # Reset
    data_df["image_vector"] = np.random.randn(3, EMBEDDING_SIZE).tolist()

    # Check embedding_vector of booleans are not allowed
    data_df["sentence_vector"] = np.random.choice(
        a=[True, False], size=(3, EMBEDDING_SIZE)
    ).tolist()
    try:
        _ = log_dataframe(data_df)
    except Exception as e:
        assert isinstance(e, err.ValidationFailure)
    # Reset
    data_df["sentence_vector"] = np.random.randn(3, EMBEDDING_SIZE).tolist()

    # Check embedding_data of float are not allowed
    data_df["sentence_data"] = [x for x in range(3)]
    try:
        _ = log_dataframe(data_df)
    except Exception as e:
        assert isinstance(e, err.ValidationFailure)
    # Reset
    data_df["sentence_data"] = ["data_" + str(x) for x in range(3)]

    # Check embedding_link_to_data of list of strings are not allowed
    data_df["image_link"] = [["link_"] + [str(x)] for x in range(3)]
    try:
        _ = log_dataframe(data_df)
    except Exception as e:
        assert isinstance(e, err.ValidationFailure)
    # Reset
    data_df["image_link"] = ["link_" + str(x) for x in range(3)]

    # Check embedding_link_to_data of float are not allowed
    data_df["image_link"] = [x for x in range(3)]
    try:
        _ = log_dataframe(data_df)
    except Exception as e:
        assert isinstance(e, err.ValidationFailure)
    # Reset
    data_df["image_link"] = ["link_" + str(x) for x in range(3)]

    # Check all resets were successful
    try:
        response = log_dataframe(data_df)
    except err.ValidationFailure:
        assert False

    # use json here because some row elements are lists and are not readily comparable
    assert (
        response.df.sort_index(axis=1).to_json()
        == roundtrip_df(data_df).sort_index(axis=1).to_json()
    )


def test_production_wrong_embedding_values():
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

    data_df["good_vector"] = good_vector
    data_df["multidimensional_vector"] = multidimensional_vector
    data_df["empty_vector"] = empty_vector
    data_df["one_vector"] = one_vector

    schema = Schema(
        prediction_id_column_name="prediction_id",
        timestamp_column_name="prediction_timestamp",
        feature_column_names=list("ABCDEFGHI"),
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
        tag_column_names=list("ABCDEFGHI"),
        prediction_label_column_name="prediction_label",
        actual_label_column_name="actual_label",
        prediction_score_column_name="prediction_score",
        actual_score_column_name="actual_score",
        shap_values_column_names=dict(zip("ABCDEF", "abcdef")),
    )

    try:
        _ = log_dataframe(data_df, schema)
    except Exception as e:
        assert isinstance(e, err.ValidationFailure)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
