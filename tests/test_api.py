import pytest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from google.protobuf.wrappers_pb2 import StringValue

import arize.public_pb2 as public__pb2
from arize.utils.types import ModelTypes
from arize.api import Client, Embedding

NUM_VAL = 20.20
STR_VAL = "arize"
BOOL_VAL = True
INT_VAL = 0
NP_FLOAT = float(1.2)
file_to_open = Path(__file__).parent / "fixtures/mpg.csv"

inputs = {
    "model_id": "model_v0",
    "model_version": "v1.2.3.4",
    "batch": "batch1234",
    "api_key": "API_KEY",
    "prediction_id": "prediction_0",
    "value_binary": BOOL_VAL,
    "value_categorical": STR_VAL,
    "value_numeric": NUM_VAL,
    "space_key": "test_space",
    "features": {
        "feature_str": STR_VAL,
        "feature_double": NUM_VAL,
        "feature_int": INT_VAL,
        "feature_bool": BOOL_VAL,
        "feature_None": None,
    },
    "embedding_features": {
        "image_embedding": Embedding(
            vector=np.array([1.0, 2, 3]),
            link_to_data="https://my-bucket.s3.us-west-2.amazonaws.com/puppy.png",
        ),
        "nlp_embedding_sentence": Embedding(
            vector=pd.Series([4.0, 5.0, 6.0, 7.0]),
            data="This is a test sentence",
        ),
        "nlp_embedding_tokens": Embedding(
            vector=pd.Series([4.0, 5.0, 6.0, 7.0]),
            data=["This", "is", "a", "test", "sentence"],
        ),
    },
    "tags": {
        "tag_str": STR_VAL,
        "tag_double": NUM_VAL,
        "tag_int": INT_VAL,
        "tag_bool": BOOL_VAL,
        "tag_None": None,
    },
    "feature_importances": {
        "feature_str": NUM_VAL,
        "feature_double": NUM_VAL,
        "feature_int": NUM_VAL,
        "feature_bool": NUM_VAL,
        "feature_numpy_float": NP_FLOAT,
    },
}


def _build_expected_record(
    p: public__pb2.Prediction = None,
    a: public__pb2.Actual = None,
    fi: public__pb2.FeatureImportances = None,
) -> public__pb2.Record:
    return public__pb2.Record(
        space_key=inputs["space_key"],
        model_id=inputs["model_id"],
        prediction_id=str(inputs["prediction_id"]),
        prediction=p,
        actual=a,
        feature_importances=fi,
    )


def _build_basic_prediction(type: str) -> public__pb2.Prediction:
    if type == "binary":
        return public__pb2.Prediction(
            label=public__pb2.Label(binary=inputs["value_binary"]),
            model_version=inputs["model_version"],
        )
    elif type == "numeric":
        return public__pb2.Prediction(
            label=public__pb2.Label(numeric=inputs["value_numeric"]),
            model_version=inputs["model_version"],
        )
    elif type == "categorical":
        return public__pb2.Prediction(
            label=public__pb2.Label(categorical=inputs["value_categorical"]),
            model_version=inputs["model_version"],
        )
    elif type == "score_categorical":
        sc = public__pb2.ScoreCategorical()
        sc.score_category.category = STR_VAL
        sc.score_category.score = NUM_VAL

        return public__pb2.Prediction(
            label=public__pb2.Label(score_categorical=sc),
            model_version=inputs["model_version"],
        )
    else:
        return public__pb2.Prediction()


def _build_basic_actual(type: str) -> public__pb2.Actual:
    if type == "binary":
        return public__pb2.Actual(
            label=public__pb2.Label(binary=inputs["value_binary"]),
        )
    elif type == "numeric":
        return public__pb2.Actual(
            label=public__pb2.Label(numeric=inputs["value_numeric"]),
        )
    elif type == "categorical":
        return public__pb2.Actual(
            label=public__pb2.Label(categorical=inputs["value_categorical"]),
        )
    elif type == "score_categorical":
        sc = public__pb2.ScoreCategorical()
        sc.category.category = STR_VAL
        return public__pb2.Actual(
            label=public__pb2.Label(score_categorical=sc),
        )
    else:
        return public__pb2.Actual()


def _attach_features_to_prediction() -> public__pb2.Prediction:
    features = {
        "feature_str": public__pb2.Value(string=STR_VAL),
        "feature_double": public__pb2.Value(double=NUM_VAL),
        "feature_int": public__pb2.Value(int=INT_VAL),
        "feature_bool": public__pb2.Value(string=str(BOOL_VAL)),
    }
    return public__pb2.Prediction(features=features)


def _attach_embedding_features_to_prediction() -> public__pb2.Prediction:
    input_embeddings = inputs["embedding_features"]
    embedding_features = {
        "image_embedding": public__pb2.Value(
            embedding=public__pb2.Embedding(
                vector=input_embeddings["image_embedding"].vector,
                link_to_data=StringValue(
                    value=input_embeddings["image_embedding"].link_to_data
                ),
            )
        ),
        "nlp_embedding_sentence": public__pb2.Value(
            embedding=public__pb2.Embedding(
                vector=input_embeddings["nlp_embedding_sentence"].vector,
                raw_data=public__pb2.Embedding.RawData(
                    tokenArray=public__pb2.Embedding.TokenArray(
                        tokens=[
                            input_embeddings["nlp_embedding_sentence"].data
                        ],  # List of a single string
                    )
                ),
                link_to_data=StringValue(
                    value=input_embeddings["nlp_embedding_sentence"].link_to_data
                ),
            )
        ),
        "nlp_embedding_tokens": public__pb2.Value(
            embedding=public__pb2.Embedding(
                vector=input_embeddings["nlp_embedding_tokens"].vector,
                raw_data=public__pb2.Embedding.RawData(
                    tokenArray=public__pb2.Embedding.TokenArray(
                        tokens=input_embeddings["nlp_embedding_tokens"].data
                    )
                ),
                link_to_data=StringValue(
                    value=input_embeddings["nlp_embedding_tokens"].link_to_data
                ),
            )
        ),
    }
    return public__pb2.Prediction(features=embedding_features)


def _attach_tags_to_prediction() -> public__pb2.Prediction:
    tags = {
        "tag_str": public__pb2.Value(string=STR_VAL),
        "tag_double": public__pb2.Value(double=NUM_VAL),
        "tag_int": public__pb2.Value(int=INT_VAL),
        "tag_bool": public__pb2.Value(string=str(BOOL_VAL)),
    }
    return public__pb2.Prediction(tags=tags)


def _attach_tags_to_actual() -> public__pb2.Actual:
    tags = {
        "tag_str": public__pb2.Value(string=STR_VAL),
        "tag_double": public__pb2.Value(double=NUM_VAL),
        "tag_int": public__pb2.Value(int=INT_VAL),
        "tag_bool": public__pb2.Value(string=str(BOOL_VAL)),
    }
    return public__pb2.Actual(tags=tags)


def mock_dataframes_clean_nan(file):
    features, tags, labels, ids = mock_dataframes(file)
    features = features.fillna("backfill")
    tags = tags.fillna("backfill")
    return features, tags, labels, ids


def mock_dataframes(file):
    features = pd.read_csv(file)
    tags = pd.read_csv(file)
    labels = pd.DataFrame(np.random.randint(1, 100, size=(features.shape[0], 1)))
    ids = pd.DataFrame([str(uuid.uuid4()) for _ in range(len(labels.index))])
    return features, tags, labels, ids


def mock_series(file):
    features = pd.read_csv(file)
    tags = pd.read_csv(file)
    labels = pd.Series(np.random.randint(1, 100, size=features.shape[0]))
    ids = pd.Series([str(uuid.uuid4()) for _ in range(len(labels.index))])
    return features, tags, labels, ids


def get_stubbed_client():
    c = Client(space_key="test_space", api_key="API_KEY", uri="https://localhost:443")

    def _post(record, uri, indexes):
        return record

    def _post_bulk(records, uri):
        return records

    def _post_preprod(records):
        return records

    c._post = _post
    c._post_bulk = _post_bulk
    c._post_preprod = _post_preprod
    return c


# TODO for each existing test that has been modified to call Client.log, add a call
# to the pre-existing method that should map to the identical cacll to Client.log to
# assert that they are equivalent


def test_build_binary_prediction_features():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_type=ModelTypes.BINARY,
        model_version=inputs["model_version"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_binary"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("binary")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p)
    #   Check result is as expected
    assert record == expected_record
    assert record.prediction.timestamp.seconds == 0
    assert record.prediction.timestamp.nanos == 0


def test_numeric_prediction_id():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_type=ModelTypes.BINARY,
        model_version=inputs["model_version"],
        prediction_id=12345,
        prediction_label=inputs["value_binary"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
        prediction_timestamp=None,
    )
    assert record.prediction_id == "12345"
    record = c.log(
        model_id=inputs["model_id"],
        model_type=ModelTypes.BINARY,
        model_version=inputs["model_version"],
        prediction_id=1.2345,
        prediction_label=inputs["value_binary"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
        prediction_timestamp=None,
    )
    assert record.prediction_id == "1.2345"


def test_numeric_dimension_name():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_type=ModelTypes.BINARY,
        model_version=inputs["model_version"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_binary"],
        features={1: "hello", 2.0: "world"},
        embedding_features={
            3: inputs["embedding_features"]["image_embedding"],
            4.0: inputs["embedding_features"]["nlp_embedding_sentence"],
        },
        tags={1: "hello", 2.0: "world"},
        prediction_timestamp=None,
    )
    for feature in record.prediction.features:
        assert isinstance(record.prediction.features[feature], public__pb2.Value)
    for tag in record.prediction.tags:
        assert isinstance(record.prediction.tags[tag], public__pb2.Value)


def test_build_binary_prediction_zero_ones():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_type=ModelTypes.BINARY,
        model_version=inputs["model_version"],
        prediction_id=inputs["prediction_id"],
        prediction_label=1,
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("binary")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p)
    #   Check result is as expected
    assert record == expected_record
    assert record.prediction.timestamp.seconds == 0
    assert record.prediction.timestamp.nanos == 0


def test_build_categorical_prediction():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_categorical"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("categorical")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p)
    #   Check result is as expected
    assert record == expected_record


def test_build_scored_prediction():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_type=ModelTypes.SCORE_CATEGORICAL,
        model_version=inputs["model_version"],
        prediction_id=inputs["prediction_id"],
        prediction_label=(inputs["value_categorical"], inputs["value_numeric"]),
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("score_categorical")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p)
    #   Check result is as expected
    assert record == expected_record


def test_build_numeric_prediction():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_numeric"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p)
    #   Check result is as expected
    assert record == expected_record


def test_build_prediction_no_embedding_features():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_numeric"],
        features=inputs["features"],
        tags=inputs["tags"],
    )

    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p)
    #   Check result is as expected
    assert record == expected_record


# Structured features refer to any feature that is not an embedding
def test_build_prediction_no_structured_features():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_numeric"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p)
    #   Check result is as expected
    assert record == expected_record


def test_build_prediction_no_features():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_numeric"],
        tags=inputs["tags"],
    )

    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_tags_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p)
    #   Check result is as expected
    assert record == expected_record


def test_build_prediction_no_tags():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_numeric"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
    )

    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p)
    #   Check result is as expected
    assert record == expected_record


def test_build_prediction_no_tags_no_features():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_numeric"],
    )

    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric")
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p)
    #   Check result is as expected
    assert record == expected_record


def test_build_scored_actual():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_type=ModelTypes.SCORE_CATEGORICAL,
        prediction_id=inputs["prediction_id"],
        actual_label=inputs["value_categorical"],
    )

    #   Start constructing expected result by building the prediction
    a = _build_basic_actual("score_categorical")
    #   Build expected record using built prediction
    expected_record = _build_expected_record(a=a)
    #   Check result is as expected
    assert record == expected_record


def test_build_numeric_actual():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        prediction_id=inputs["prediction_id"],
        actual_label=inputs["value_numeric"],
    )
    #   Start constructing expected result by building the prediction
    a = _build_basic_actual("numeric")
    #   Build expected record using built prediction
    expected_record = _build_expected_record(a=a)
    #   Check result is as expected
    assert record == expected_record
    assert record.actual.timestamp.seconds == 0
    assert record.actual.timestamp.nanos == 0


def test_build_categorical_actual():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        prediction_id=inputs["prediction_id"],
        actual_label=inputs["value_categorical"],
    )
    #   Start constructing expected result by building the prediction
    a = _build_basic_actual("categorical")
    #   Build expected record using built prediction
    expected_record = _build_expected_record(a=a)
    #   Check result is as expected
    assert record == expected_record


def test_build_binary_actual():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        prediction_id=inputs["prediction_id"],
        actual_label=inputs["value_binary"],
    )
    #   Start constructing expected result by building the prediction
    a = _build_basic_actual("binary")
    #   Build expected record using built prediction
    expected_record = _build_expected_record(a=a)
    #   Check result is as expected
    assert record == expected_record


def test_build_prediction_and_actual_numeric():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_numeric"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
        actual_label=inputs["value_numeric"],
    )
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric")
    a = _build_basic_actual("numeric")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #  Add tags to actual
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, a=a)
    #   Check result is as expected
    assert record == expected_record


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
