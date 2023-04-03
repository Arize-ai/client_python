from pathlib import Path

import arize.public_pb2 as pb2
import numpy as np
import pandas as pd
import pytest
from arize.api import Client, Embedding
from arize.utils.types import (
    Environments,
    ModelTypes,
    ObjectDetectionLabel,
    RankingActualLabel,
    RankingPredictionLabel,
)
from google.protobuf.wrappers_pb2 import DoubleValue, StringValue

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
    "model_type_object_detection": ModelTypes.OBJECT_DETECTION,
    "model_type_ranking": ModelTypes.RANKING,
    "model_version": "v1.2.3.4",
    "environment_training": Environments.TRAINING,
    "environment_validation": Environments.VALIDATION,
    "environment_production": Environments.PRODUCTION,
    "batch_id": "batch_id",
    "batch": "batch1234",
    "api_key": "API_KEY",
    "prediction_id": "prediction_0",
    "label_bool": BOOL_VAL,
    "label_str": STR_VAL,
    "label_int": INT_VAL,
    "label_float": FLOAT_VAL,
    "label_tuple": (STR_VAL, FLOAT_VAL),
    "object_detection_bounding_boxes": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
    "object_detection_categories": ["dog", "cat"],
    "object_detection_scores": [0.8, 0.4],
    "ranking_group_id": "a",
    "ranking_rank": 1,
    "ranking_prediction_score": 1.0,
    "ranking_label": "click",
    "ranking_relevance_labels": ["click", "save"],
    "ranking_relevance_score": 0.5,
    "space_key": "test_space",
    "features": {
        "feature_str": STR_VAL,
        "feature_double": FLOAT_VAL,
        "feature_int": INT_VAL,
        "feature_bool": BOOL_VAL,
        "feature_None": None,
    },
    "object_detection_embedding_feature": {
        "image_embedding": Embedding(
            vector=np.array([1.0, 2, 3]),
            link_to_data="https://my-bucket.s3.us-west-2.amazonaws.com/puppy.png",
        ),
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
    ep: pb2.Record.EnvironmentParams = None,
    p: pb2.PredictionLabel = None,
    a: pb2.ActualLabel = None,
    fi: pb2.FeatureImportances = None,
) -> pb2.Record:
    return pb2.Record(
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
) -> pb2.Record.EnvironmentParams:
    env_params = None
    if env == Environments.TRAINING:
        env_params = pb2.Record.EnvironmentParams(training=pb2.Record.EnvironmentParams.Training())
    elif env == Environments.VALIDATION:
        env_params = pb2.Record.EnvironmentParams(
            validation=pb2.Record.EnvironmentParams.Validation(batch_id=inputs["batch_id"])
        )
    elif env == Environments.PRODUCTION:
        env_params = pb2.Record.EnvironmentParams(
            production=pb2.Record.EnvironmentParams.Production()
        )
    return env_params


def _build_basic_prediction(type: str) -> pb2.Prediction:
    if type == "numeric_int":
        return pb2.Prediction(
            prediction_label=pb2.PredictionLabel(numeric=inputs["label_int"]),
            model_version=inputs["model_version"],
        )
    elif type == "numeric_float":
        return pb2.Prediction(
            prediction_label=pb2.PredictionLabel(numeric=inputs["label_float"]),
            model_version=inputs["model_version"],
        )
    elif type == "score_categorical_bool":
        sc = pb2.ScoreCategorical()
        sc.category.category = str(inputs["label_bool"])
        return pb2.Prediction(
            prediction_label=pb2.PredictionLabel(score_categorical=sc),
            model_version=inputs["model_version"],
        )
    elif type == "score_categorical_str":
        sc = pb2.ScoreCategorical()
        sc.category.category = inputs["label_str"]
        return pb2.Prediction(
            prediction_label=pb2.PredictionLabel(score_categorical=sc),
            model_version=inputs["model_version"],
        )
    elif type == "score_categorical_tuple":
        sc = pb2.ScoreCategorical()
        sc.score_category.category = inputs["label_str"]
        sc.score_category.score = inputs["label_float"]
        return pb2.Prediction(
            prediction_label=pb2.PredictionLabel(score_categorical=sc),
            model_version=inputs["model_version"],
        )
    elif type == "object_detection":
        od = pb2.ObjectDetection()
        bounding_boxes = []
        for i in range(len(inputs["object_detection_bounding_boxes"])):
            coordinates = inputs["object_detection_bounding_boxes"][i]
            category = inputs["object_detection_categories"][i]
            score = inputs["object_detection_scores"][i]
            bounding_boxes.append(
                pb2.ObjectDetection.BoundingBox(
                    coordinates=coordinates, category=category, score=DoubleValue(value=score)
                )
            )
        od.bounding_boxes.extend(bounding_boxes)
        return pb2.Prediction(
            prediction_label=pb2.PredictionLabel(object_detection=od),
            model_version=inputs["model_version"],
        )
    elif type == "ranking":
        rp = pb2.RankingPrediction()
        rp.rank = inputs["ranking_rank"]
        rp.prediction_group_id = inputs["ranking_group_id"]
        rp.prediction_score.value = inputs["ranking_prediction_score"]
        rp.label = inputs["ranking_label"]
        return pb2.Prediction(
            prediction_label=pb2.PredictionLabel(ranking=rp),
            model_version=inputs["model_version"],
        )
    else:
        return pb2.Prediction()


def _build_basic_actual(type: str) -> pb2.Actual:
    if type == "numeric_int":
        return pb2.Actual(
            actual_label=pb2.ActualLabel(numeric=inputs["label_int"]),
        )
    elif type == "numeric_float":
        return pb2.Actual(
            actual_label=pb2.ActualLabel(numeric=inputs["label_float"]),
        )
    elif type == "score_categorical_bool":
        sc = pb2.ScoreCategorical()
        sc.category.category = str(inputs["label_bool"])
        return pb2.Actual(
            actual_label=pb2.ActualLabel(score_categorical=sc),
        )
    elif type == "score_categorical_str":
        sc = pb2.ScoreCategorical()
        sc.category.category = inputs["label_str"]
        return pb2.Actual(
            actual_label=pb2.ActualLabel(score_categorical=sc),
        )
    elif type == "score_categorical_tuple":
        sc = pb2.ScoreCategorical()
        sc.score_category.category = inputs["label_str"]
        sc.score_category.score = inputs["label_float"]
        return pb2.Actual(
            actual_label=pb2.ActualLabel(score_categorical=sc),
        )
    elif type == "object_detection":
        od = pb2.ObjectDetection()
        bounding_boxes = []
        for i in range(len(inputs["object_detection_bounding_boxes"])):
            coordinates = inputs["object_detection_bounding_boxes"][i]
            category = inputs["object_detection_categories"][i]
            bounding_boxes.append(
                pb2.ObjectDetection.BoundingBox(coordinates=coordinates, category=category)
            )
        od.bounding_boxes.extend(bounding_boxes)
        return pb2.Actual(
            actual_label=pb2.ActualLabel(object_detection=od),
        )
    elif type == "ranking":
        ra = pb2.RankingActual()
        ra.category.values.extend(inputs["ranking_relevance_labels"])
        ra.relevance_score.value = inputs["ranking_relevance_score"]
        return pb2.Actual(
            actual_label=pb2.ActualLabel(ranking=ra),
        )
    else:
        return pb2.Actual()


def _attach_features_to_prediction() -> pb2.Prediction:
    features = {
        "feature_str": pb2.Value(string=STR_VAL),
        "feature_double": pb2.Value(double=FLOAT_VAL),
        "feature_int": pb2.Value(int=INT_VAL),
        "feature_bool": pb2.Value(string=str(BOOL_VAL)),
    }
    return pb2.Prediction(features=features)


def _attach_image_embedding_feature_to_prediction() -> pb2.Prediction:
    input_embeddings = inputs["embedding_features"]
    embedding_features = {
        "image_embedding": pb2.Value(
            embedding=pb2.Embedding(
                vector=input_embeddings["image_embedding"].vector,
                link_to_data=StringValue(value=input_embeddings["image_embedding"].link_to_data),
            )
        ),
    }
    return pb2.Prediction(features=embedding_features)


def _attach_embedding_features_to_prediction() -> pb2.Prediction:
    input_embeddings = inputs["embedding_features"]
    embedding_features = {
        "image_embedding": pb2.Value(
            embedding=pb2.Embedding(
                vector=input_embeddings["image_embedding"].vector,
                link_to_data=StringValue(value=input_embeddings["image_embedding"].link_to_data),
            )
        ),
        "nlp_embedding_sentence": pb2.Value(
            embedding=pb2.Embedding(
                vector=input_embeddings["nlp_embedding_sentence"].vector,
                raw_data=pb2.Embedding.RawData(
                    tokenArray=pb2.Embedding.TokenArray(
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
        "nlp_embedding_tokens": pb2.Value(
            embedding=pb2.Embedding(
                vector=input_embeddings["nlp_embedding_tokens"].vector,
                raw_data=pb2.Embedding.RawData(
                    tokenArray=pb2.Embedding.TokenArray(
                        tokens=input_embeddings["nlp_embedding_tokens"].data
                    )
                ),
                link_to_data=StringValue(
                    value=input_embeddings["nlp_embedding_tokens"].link_to_data
                ),
            )
        ),
    }
    return pb2.Prediction(features=embedding_features)


def _attach_tags_to_prediction() -> pb2.Prediction:
    tags = {
        "tag_str": pb2.Value(string=STR_VAL),
        "tag_double": pb2.Value(double=FLOAT_VAL),
        "tag_int": pb2.Value(int=INT_VAL),
        "tag_bool": pb2.Value(string=str(BOOL_VAL)),
    }
    return pb2.Prediction(tags=tags)


def _attach_tags_to_actual() -> pb2.Actual:
    tags = {
        "tag_str": pb2.Value(string=STR_VAL),
        "tag_double": pb2.Value(double=FLOAT_VAL),
        "tag_int": pb2.Value(int=INT_VAL),
        "tag_bool": pb2.Value(string=str(BOOL_VAL)),
    }
    return pb2.Actual(tags=tags)


def get_stubbed_client():
    c = Client(space_key="test_space", api_key="API_KEY", uri="https://localhost:443")

    def _post(record, uri, indexes):
        return record

    c._post = _post
    return c


# TODO for each existing test that has been modified to call Client.log, add a call
# to the pre-existing method that should map to the identical call to Client.log to
# assert that they are equivalent


def test_build_pred_and_actual_label_bool():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_score_categorical"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_bool"],
        actual_label=inputs["label_bool"],
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
        prediction_label=inputs["label_str"],
        actual_label=inputs["label_str"],
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
        prediction_label=inputs["label_int"],
        actual_label=inputs["label_int"],
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
        prediction_label=inputs["label_int"],
        actual_label=inputs["label_int"],
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
        prediction_label=inputs["label_float"],
        actual_label=inputs["label_float"],
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
        prediction_label=inputs["label_float"],
        actual_label=inputs["label_float"],
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
        prediction_label=inputs["label_tuple"],
        actual_label=inputs["label_tuple"],
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
        prediction_label=inputs["label_tuple"],
        actual_label=inputs["label_tuple"],
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


def test_build_pred_and_actual_label_ranking():
    pred_label = RankingPredictionLabel(
        group_id=inputs["ranking_group_id"],
        rank=inputs["ranking_rank"],
        score=inputs["ranking_prediction_score"],
        label=inputs["ranking_label"],
    )
    act_label = RankingActualLabel(
        relevance_labels=inputs["ranking_relevance_labels"],
        relevance_score=inputs["ranking_relevance_score"],
    )
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_ranking"],
        prediction_id=inputs["prediction_id"],
        prediction_label=pred_label,
        actual_label=act_label,
        features=inputs["features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("ranking")
    a = _build_basic_actual("ranking")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Add props to prediction according to this test
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, a=a, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_ranking_label_missing_group_id_rank():
    with pytest.raises(TypeError) as excinfo:
        _ = RankingPredictionLabel(
            group_id=inputs["ranking_group_id"],
            score=inputs["ranking_prediction_score"],
            label=inputs["ranking_label"],
        )
    assert "missing 1 required positional argument: 'rank'" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        _ = RankingPredictionLabel(
            rank=inputs["ranking_rank"],
            score=inputs["ranking_prediction_score"],
            label=inputs["ranking_label"],
        )
    assert "missing 1 required positional argument: 'group_id'" in str(excinfo.value)


def test_build_wrong_ranking_rank():
    c = get_stubbed_client()
    pred_label = RankingPredictionLabel(
        group_id=inputs["ranking_group_id"],
        rank=101,
        score=inputs["ranking_prediction_score"],
        label=inputs["ranking_label"],
    )
    act_label = RankingActualLabel(
        relevance_labels=inputs["ranking_relevance_labels"],
        relevance_score=inputs["ranking_relevance_score"],
    )

    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            model_type=inputs["model_type_ranking"],
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            actual_label=act_label,
            features=inputs["features"],
            tags=inputs["tags"],
        )
    assert "Rank must be between 1 and 100, inclusive. Found 101" in str(excinfo.value)


def test_build_wrong_ranking_group_id():
    c = get_stubbed_client()
    pred_label = RankingPredictionLabel(
        group_id=1,
        rank=inputs["ranking_rank"],
        score=inputs["ranking_prediction_score"],
        label=inputs["ranking_label"],
    )
    act_label = RankingActualLabel(
        relevance_labels=inputs["ranking_relevance_labels"],
        relevance_score=inputs["ranking_relevance_score"],
    )

    with pytest.raises(TypeError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            model_type=inputs["model_type_ranking"],
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            actual_label=act_label,
            features=inputs["features"],
            tags=inputs["tags"],
        )
    assert "Prediction Group ID must be a string" in str(excinfo.value)

    pred_label = RankingPredictionLabel(
        group_id="aaabbbcccdddeeefffggghhhiiijjjkkklllmmmnnn",
        rank=inputs["ranking_rank"],
        score=inputs["ranking_prediction_score"],
        label=inputs["ranking_label"],
    )

    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            model_type=inputs["model_type_ranking"],
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            actual_label=act_label,
            features=inputs["features"],
            tags=inputs["tags"],
        )
    assert "Prediction Group ID must have length between 1 and 36. Found 42" in str(excinfo.value)


def test_build_wrong_ranking_relevance_labels():
    c = get_stubbed_client()
    pred_label = RankingPredictionLabel(
        group_id=inputs["ranking_group_id"],
        rank=inputs["ranking_rank"],
        score=inputs["ranking_prediction_score"],
        label=inputs["ranking_label"],
    )
    act_label = RankingActualLabel(
        relevance_labels=["click", ""], relevance_score=inputs["ranking_relevance_score"]
    )

    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            model_type=inputs["model_type_ranking"],
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            actual_label=act_label,
            features=inputs["features"],
            tags=inputs["tags"],
        )
    assert "Relevance Labels must be not contain empty strings" in str(excinfo.value)


def test_build_wrong_ranking_relevance_scores():
    c = get_stubbed_client()
    pred_label = RankingPredictionLabel(
        group_id=inputs["ranking_group_id"],
        rank=inputs["ranking_rank"],
        score=inputs["ranking_prediction_score"],
        label=inputs["ranking_label"],
    )
    act_label = RankingActualLabel(
        relevance_labels=inputs["ranking_relevance_labels"], relevance_score="click"
    )

    with pytest.raises(TypeError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            model_type=inputs["model_type_ranking"],
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            actual_label=act_label,
            features=inputs["features"],
            tags=inputs["tags"],
        )
    assert "Relevance score must be a float or an int" in str(excinfo.value)


def test_build_pred_and_actual_label_object_detection():
    pred_label = ObjectDetectionLabel(
        bounding_boxes_coordinates=inputs["object_detection_bounding_boxes"],
        categories=inputs["object_detection_categories"],
        scores=inputs["object_detection_scores"],
    )
    act_label = ObjectDetectionLabel(
        bounding_boxes_coordinates=inputs["object_detection_bounding_boxes"],
        categories=inputs["object_detection_categories"],
    )
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_object_detection"],
        prediction_id=inputs["prediction_id"],
        prediction_label=pred_label,
        actual_label=act_label,
        features=inputs["features"],
        embedding_features=inputs["object_detection_embedding_feature"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(inputs["environment_production"])
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("object_detection")
    a = _build_basic_actual("object_detection")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_image_embedding_feature_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Add props to prediction according to this test
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, a=a, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_build_prediction_no_embedding_features():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_numeric"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_float"],
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
        prediction_label=inputs["label_float"],
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
        prediction_label=inputs["label_float"],
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
        prediction_label=inputs["label_float"],
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
        prediction_label=inputs["label_float"],
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
            prediction_label=inputs["label_str"],
            actual_label=inputs["label_str"],
            features=inputs["features"],
            embedding_features=inputs["embedding_features"],
            tags=inputs["tags"],
        )
    assert "log() missing 1 required positional argument: 'model_type'" in str(excinfo.value)


def test_model_version_optional():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        environment=inputs["environment_production"],
        model_type=inputs["model_type_numeric"],
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_float"],
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
            prediction_label=inputs["label_str"],
            actual_label=inputs["label_str"],
            features=inputs["features"],
            embedding_features=inputs["embedding_features"],
            tags=inputs["tags"],
        )
    assert "log() missing 1 required positional argument: 'environment'" in str(excinfo.value)


def test_object_detection_item_count_match():
    c = get_stubbed_client()
    extra = [0.11, 0.12, 0.13, 0.14]

    pred_label = ObjectDetectionLabel(
        bounding_boxes_coordinates=inputs["object_detection_bounding_boxes"] + [extra],
        categories=inputs["object_detection_categories"],
        scores=inputs["object_detection_scores"],
    )
    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            model_type=inputs["model_type_object_detection"],
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            features=inputs["features"],
            embedding_features=inputs["object_detection_embedding_feature"],
            tags=inputs["tags"],
        )
    assert (
        "Object Detection Labels must contain the same number of bounding boxes and "
        "categories. Found 3 bounding boxes and 2 categories." in str(excinfo.value)
    )


def test_object_detection_wrong_coordinates_format():
    c = get_stubbed_client()

    pred_label = ObjectDetectionLabel(
        bounding_boxes_coordinates=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7]],
        categories=inputs["object_detection_categories"],
        scores=inputs["object_detection_scores"],
    )
    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            model_type=inputs["model_type_object_detection"],
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            features=inputs["features"],
            embedding_features=inputs["object_detection_embedding_feature"],
            tags=inputs["tags"],
        )
    assert "Each bounding box's coordinates must be a collection of 4 floats." in str(excinfo.value)

    pred_label = ObjectDetectionLabel(
        bounding_boxes_coordinates=[[-0.1, 0.2, 0.3, 0.4], [1.5, 0.6, 0.7, 0.8]],
        categories=inputs["object_detection_categories"],
        scores=inputs["object_detection_scores"],
    )
    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            model_type=inputs["model_type_object_detection"],
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            features=inputs["features"],
            embedding_features=inputs["object_detection_embedding_feature"],
            tags=inputs["tags"],
        )
    assert "Bounding box's coordinates cannot be negative. Found [-0.1, 0.2, 0.3, 0.4]" in str(
        excinfo.value
    )


def test_valid_prediction_id_embeddings():
    c = get_stubbed_client()

    # test case - too long prediction_id
    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_type=inputs["model_type_binary_classification"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            prediction_id="A" * 129,
            prediction_label=inputs["label_str"],
            actual_label=inputs["label_str"],
            features=inputs["features"],
            embedding_features=inputs["embedding_features"],
            tags=inputs["tags"],
        )
    assert "The string length of prediction_id" in str(excinfo.value)


def test_no_prediction_id():
    c = get_stubbed_client()

    # test case - None prediction_id
    try:
        _ = c.log(
            model_id=inputs["model_id"],
            model_type=inputs["model_type_binary_classification"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            prediction_id=None,
            prediction_label=inputs["label_str"],
            actual_label=inputs["label_str"],
            features=inputs["features"],
            embedding_features=inputs["embedding_features"],
            tags=inputs["tags"],
        )
    except Exception as e:
        assert False, f"Logging data without prediction_id raised an exception {e}"


def test_object_detection_wrong_categories():
    c = get_stubbed_client()

    pred_label = ObjectDetectionLabel(
        bounding_boxes_coordinates=inputs["object_detection_bounding_boxes"],
        categories=["dog", None],
        scores=inputs["object_detection_scores"],
    )
    with pytest.raises(TypeError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            model_type=inputs["model_type_object_detection"],
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            features=inputs["features"],
            embedding_features=inputs["object_detection_embedding_feature"],
            tags=inputs["tags"],
        )
    assert "Object Detection Label categories must be a list of strings" in str(excinfo.value)


def test_object_detection_wrong_scores():
    c = get_stubbed_client()

    pred_label = ObjectDetectionLabel(
        bounding_boxes_coordinates=inputs["object_detection_bounding_boxes"],
        categories=inputs["object_detection_categories"],
        scores=[-0.4],
    )
    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            model_type=inputs["model_type_object_detection"],
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            features=inputs["features"],
            embedding_features=inputs["object_detection_embedding_feature"],
            tags=inputs["tags"],
        )
    assert "Bounding box confidence scores must be between 0 and 1, inclusive" in str(excinfo.value)

    pred_label = ObjectDetectionLabel(
        bounding_boxes_coordinates=inputs["object_detection_bounding_boxes"],
        categories=inputs["object_detection_categories"],
        scores=[1.2],
    )
    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=inputs["environment_production"],
            model_type=inputs["model_type_object_detection"],
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            features=inputs["features"],
            embedding_features=inputs["object_detection_embedding_feature"],
            tags=inputs["tags"],
        )
    assert "Bounding box confidence scores must be between 0 and 1, inclusive" in str(excinfo.value)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
