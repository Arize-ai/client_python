from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from google.protobuf.wrappers_pb2 import StringValue

import arize.public_pb2 as public__pb2
from arize.api import Client, Embedding
from arize.utils.types import ModelTypes, Environments

BOOL_VAL = True
STR_VAL = "arize"
INT_VAL = 5
FLOAT_VAL = 20.20
NP_FLOAT = float(1.2)
file_to_open = Path(__file__).parent / "fixtures/mpg.csv"

inputs = {
    "model_id": "model_v0",
    "model_version": "v1.2.3.4",
    "model_type_numeric": ModelTypes.NUMERIC,
    "model_type_score_categorical": ModelTypes.SCORE_CATEGORICAL,
    "model_type_regression": ModelTypes.REGRESSION,
    "model_type_binary_classification": ModelTypes.BINARY_CLASSIFICATION,
    "model_version": "v1.2.3.4",
    "environment_training": Environments.TRAINING,
    "environment_validation": Environments.VALIDATION,
    "environment_production": Environments.PRODUCTION,
    "batch_id": "batch_id",
    "batch": "batch1234",
    "api_key": "API_KEY",
    "prediction_id": "prediction_0",
    "value_bool": BOOL_VAL,
    "value_str": STR_VAL,
    "value_int": INT_VAL,
    "value_float": FLOAT_VAL,
    "value_tuple": (STR_VAL, FLOAT_VAL),
    "space_key": "test_space",
    "features": {
        "feature_str": STR_VAL,
        "feature_double": FLOAT_VAL,
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
        "tag_double": FLOAT_VAL,
        "tag_int": INT_VAL,
        "tag_bool": BOOL_VAL,
        "tag_None": None,
    },
    "feature_importances": {
        "feature_str": FLOAT_VAL,
        "feature_double": FLOAT_VAL,
        "feature_int": INT_VAL,
        "feature_bool": BOOL_VAL,
        "feature_numpy_float": NP_FLOAT,
    },
}


def _build_expected_record(
    ep: public__pb2.Record.EnvironmentParams = None,
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
        environment_params=ep,
    )


def _get_proto_environment_params(
    env: Environments,
) -> public__pb2.Record.EnvironmentParams:
    env_params = None
    if env == Environments.TRAINING:
        env_params = public__pb2.Record.EnvironmentParams(
            training=public__pb2.Record.EnvironmentParams.Training()
        )
    elif env == Environments.VALIDATION:
        env_params = public__pb2.Record.EnvironmentParams(
            validation=public__pb2.Record.EnvironmentParams.Validation(
                batch_id=inputs["batch_id"]
            )
        )
    elif env == Environments.PRODUCTION:
        env_params = public__pb2.Record.EnvironmentParams(
            production=public__pb2.Record.EnvironmentParams.Production()
        )
    return env_params


def _build_basic_prediction(type: str) -> public__pb2.Prediction:
    if type == "numeric_int":
        return public__pb2.Prediction(
            label=public__pb2.Label(numeric=inputs["value_int"]),
            model_version=inputs["model_version"],
        )
    elif type == "numeric_float":
        return public__pb2.Prediction(
            label=public__pb2.Label(numeric=inputs["value_float"]),
            model_version=inputs["model_version"],
        )
    elif type == "score_categorical_bool":
        sc = public__pb2.ScoreCategorical()
        sc.category.category = str(inputs["value_bool"])
        return public__pb2.Prediction(
            label=public__pb2.Label(score_categorical=sc),
            model_version=inputs["model_version"],
        )
    elif type == "score_categorical_str":
        sc = public__pb2.ScoreCategorical()
        sc.category.category = inputs["value_str"]
        return public__pb2.Prediction(
            label=public__pb2.Label(score_categorical=sc),
            model_version=inputs["model_version"],
        )
    elif type == "score_categorical_tuple":
        sc = public__pb2.ScoreCategorical()
        sc.score_category.category = inputs["value_str"]
        sc.score_category.score = inputs["value_float"]
        return public__pb2.Prediction(
            label=public__pb2.Label(score_categorical=sc),
            model_version=inputs["model_version"],
        )
    else:
        return public__pb2.Prediction()


def _build_basic_actual(type: str) -> public__pb2.Actual:
    if type == "numeric_int":
        return public__pb2.Actual(
            label=public__pb2.Label(numeric=inputs["value_int"]),
        )
    elif type == "numeric_float":
        return public__pb2.Actual(
            label=public__pb2.Label(numeric=inputs["value_float"]),
        )
    elif type == "score_categorical_bool":
        sc = public__pb2.ScoreCategorical()
        sc.category.category = str(inputs["value_bool"])
        return public__pb2.Actual(
            label=public__pb2.Label(score_categorical=sc),
        )
    elif type == "score_categorical_str":
        sc = public__pb2.ScoreCategorical()
        sc.category.category = inputs["value_str"]
        return public__pb2.Actual(
            label=public__pb2.Label(score_categorical=sc),
        )
    elif type == "score_categorical_tuple":
        sc = public__pb2.ScoreCategorical()
        sc.score_category.category = inputs["value_str"]
        sc.score_category.score = inputs["value_float"]
        return public__pb2.Actual(
            label=public__pb2.Label(score_categorical=sc),
        )
    else:
        return public__pb2.Actual()


def _attach_features_to_prediction() -> public__pb2.Prediction:
    features = {
        "feature_str": public__pb2.Value(string=STR_VAL),
        "feature_double": public__pb2.Value(double=FLOAT_VAL),
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
        "tag_double": public__pb2.Value(double=FLOAT_VAL),
        "tag_int": public__pb2.Value(int=INT_VAL),
        "tag_bool": public__pb2.Value(string=str(BOOL_VAL)),
    }
    return public__pb2.Prediction(tags=tags)


def _attach_tags_to_actual() -> public__pb2.Actual:
    tags = {
        "tag_str": public__pb2.Value(string=STR_VAL),
        "tag_double": public__pb2.Value(double=FLOAT_VAL),
        "tag_int": public__pb2.Value(int=INT_VAL),
        "tag_bool": public__pb2.Value(string=str(BOOL_VAL)),
    }
    return public__pb2.Actual(tags=tags)


def get_stubbed_client():
    c = Client(space_key="test_space", api_key="API_KEY", uri="https://localhost:443")

    def _post(record, uri, indexes):
        return record

    c._post = _post
    return c


# TODO for each existing test that has been modified to call Client.log, add a call
# to the pre-existing method that should map to the identical cacll to Client.log to
# assert that they are equivalent


def test_build_pred_and_actual_label_bool():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_score_categorical"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_bool"],
        actual_label=inputs["value_bool"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("score_categorical_bool")
    a = _build_basic_actual("score_categorical_bool")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Add props to prediction according to this test
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, a=a, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_build_pred_and_actual_label_str():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_score_categorical"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_str"],
        actual_label=inputs["value_str"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("score_categorical_str")
    a = _build_basic_actual("score_categorical_str")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Add props to prediction according to this test
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, a=a, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_build_pred_and_actual_label_int():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_numeric"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_int"],
        actual_label=inputs["value_int"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    record_new_model_type = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_regression"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_int"],
        actual_label=inputs["value_int"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric_int")
    a = _build_basic_actual("numeric_int")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Add props to prediction according to this test
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, a=a, ep=ep)
    #   Check result is as expected
    assert record == record_new_model_type == expected_record


def test_build_pred_and_actual_label_float():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_numeric"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_float"],
        actual_label=inputs["value_float"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    record_new_model_type = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_regression"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_float"],
        actual_label=inputs["value_float"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric_float")
    a = _build_basic_actual("numeric_float")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Add props to prediction according to this test
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, a=a, ep=ep)
    #   Check result is as expected
    assert record == record_new_model_type == expected_record


def test_build_pred_and_actual_label_tuple():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_score_categorical"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_tuple"],
        actual_label=inputs["value_tuple"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    record_new_model_type = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_binary_classification"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_tuple"],
        actual_label=inputs["value_tuple"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("score_categorical_tuple")
    a = _build_basic_actual("score_categorical_tuple")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Add props to prediction according to this test
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, a=a, ep=ep)
    #   Check result is as expected
    assert record == record_new_model_type == expected_record


def test_build_prediction_no_embedding_features():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_numeric"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_float"],
        features=inputs["features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric_float")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, ep=ep)
    #   Check result is as expected
    assert record == expected_record


# Structured features refer to any feature that is not an embedding
def test_build_prediction_no_structured_features():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_numeric"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_float"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric_float")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_build_prediction_no_features():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_numeric"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_float"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric_float")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_tags_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_build_prediction_no_tags():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_numeric"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_float"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric_float")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_build_prediction_no_tags_no_features():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_numeric"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_float"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric_float")
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_missing_model_type():
    c = get_stubbed_client()
    with pytest.raises(TypeError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["value_str"],
            actual_label=inputs["value_str"],
            features=inputs["features"],
            embedding_features=inputs["embedding_features"],
            tags=inputs["tags"],
        )
    assert "log() missing 1 required positional argument: 'model_type'" in str(
        excinfo.value
    )


def test_model_version_optional():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_numeric"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["value_float"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric_float")
    p.model_version = ""
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_missing_environment():
    c = get_stubbed_client()
    with pytest.raises(TypeError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            model_type=inputs["model_type_numeric"],
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["value_str"],
            actual_label=inputs["value_str"],
            features=inputs["features"],
            embedding_features=inputs["embedding_features"],
            tags=inputs["tags"],
        )
    assert "log() missing 1 required positional argument: 'environment'" in str(
        excinfo.value
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
