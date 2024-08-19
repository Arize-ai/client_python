import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import arize.public_pb2 as pb2
import arize.utils.errors as err
import numpy as np
import pandas as pd
import pytest
from arize import __version__ as arize_version
from arize.api import Client
from arize.pandas.validation.errors import InvalidAdditionalHeaders, InvalidNumberOfEmbeddings
from arize.single_log.errors import CastingError
from arize.utils.constants import (
    API_KEY_ENVVAR_NAME,
    LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_NUMBER_OF_EMBEDDINGS,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
    MAX_PREDICTION_ID_LEN,
    MAX_TAG_LENGTH,
    MIN_PREDICTION_ID_LEN,
    RESERVED_TAG_COLS,
    SPACE_ID_ENVVAR_NAME,
    SPACE_KEY_ENVVAR_NAME,
)
from arize.utils.types import (
    ArizeTypes,
    Embedding,
    Environments,
    LLMRunMetadata,
    ModelTypes,
    MultiClassActualLabel,
    MultiClassPredictionLabel,
    ObjectDetectionLabel,
    RankingActualLabel,
    RankingPredictionLabel,
    TypedValue,
)
from arize.utils.utils import convert_dictionary, get_python_version
from google.protobuf.wrappers_pb2 import BoolValue, DoubleValue, StringValue

BOOL_VAL = True
STR_VAL = "arize"
INT_VAL = 5
FLOAT_VAL = 20.20
NP_FLOAT = float(1.2)
STR_LST_VAL = ["apple", "banana", "orange"]
file_to_open = Path(__file__).parent / "fixtures/mpg.csv"

inputs = {
    "space_id": "hmac_encoded_space_id",
    "model_id": "model_v0",
    "model_version": "v1.2.3.4",
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
    "multi_class_prediction_scores": {"class1": 0.2, "class2": 0.1, "class3": 0.4},
    "multi_class_threshold_scores": {"class1": 0.1, "class2": 0.2, "class3": 0.3},
    "multi_class_actual_scores": {"class1": 0, "class2": 0, "class3": 1},
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
        "feature_str_lst": STR_LST_VAL,
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
    "prompt_embedding": Embedding(
        vector=pd.Series([4.0, 5.0, 6.0, 7.0]),
        data="This is a test prompt",
    ),
    "response_embedding": Embedding(
        vector=pd.Series([4.0, 5.0, 6.0, 7.0]),
        data="This is a test response",
    ),
    "prompt_str": "This is a test prompt",
    "response_str": "This is a test response",
}


def _build_expected_record(
    ep: pb2.Record.EnvironmentParams = None,
    p: pb2.PredictionLabel = None,
    a: pb2.ActualLabel = None,
    fi: pb2.FeatureImportances = None,
    is_generative_llm_record: BoolValue = BoolValue(value=False),
) -> pb2.Record:
    return pb2.Record(
        space_key=inputs["space_key"],
        model_id=inputs["model_id"],
        prediction_id=str(inputs["prediction_id"]),
        prediction=p,
        actual=a,
        feature_importances=fi,
        environment_params=ep,
        is_generative_llm_record=is_generative_llm_record,
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
    elif type == "score_categorical_bool" or type == "generative_bool":
        sc = pb2.ScoreCategorical()
        sc.category.category = str(inputs["label_bool"])
        return pb2.Prediction(
            prediction_label=pb2.PredictionLabel(score_categorical=sc),
            model_version=inputs["model_version"],
        )
    elif type == "score_categorical_str" or type == "generative_str":
        sc = pb2.ScoreCategorical()
        sc.category.category = inputs["label_str"]
        return pb2.Prediction(
            prediction_label=pb2.PredictionLabel(score_categorical=sc),
            model_version=inputs["model_version"],
        )
    elif type == "score_categorical_int" or type == "generative_int":
        sc = pb2.ScoreCategorical()
        sc.score_value.value = inputs["label_int"]
        return pb2.Prediction(
            prediction_label=pb2.PredictionLabel(score_categorical=sc),
            model_version=inputs["model_version"],
        )
    elif type == "score_categorical_float" or type == "generative_float":
        sc = pb2.ScoreCategorical()
        sc.score_value.value = inputs["label_float"]
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
    elif type == "multi_class":
        prediction_scores = inputs["multi_class_prediction_scores"]
        threshold_scores = inputs["multi_class_threshold_scores"]
        prediction_threshold_scores = {}
        for class_name, prediction_score in prediction_scores.items():
            multi_label_scores = pb2.MultiClassPrediction.MultiLabel.MultiLabelScores(
                prediction_score=DoubleValue(value=prediction_score),
                threshold_score=DoubleValue(value=threshold_scores[class_name]),
            )
            prediction_threshold_scores[class_name] = multi_label_scores
        multi_label = pb2.MultiClassPrediction.MultiLabel(
            prediction_threshold_scores=prediction_threshold_scores,
        )
        mc_pred = pb2.MultiClassPrediction(multi_label=multi_label)
        return pb2.Prediction(
            prediction_label=pb2.PredictionLabel(multi_class=mc_pred),
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


def _build_basic_actual(type: str = "") -> pb2.Actual:
    if type == "numeric_int":
        return pb2.Actual(
            actual_label=pb2.ActualLabel(numeric=inputs["label_int"]),
        )
    elif type == "numeric_float":
        return pb2.Actual(
            actual_label=pb2.ActualLabel(numeric=inputs["label_float"]),
        )
    elif type == "score_categorical_bool" or type == "generative_bool":
        sc = pb2.ScoreCategorical()
        sc.category.category = str(inputs["label_bool"])
        return pb2.Actual(
            actual_label=pb2.ActualLabel(score_categorical=sc),
        )
    elif type == "score_categorical_str" or type == "generative_str":
        sc = pb2.ScoreCategorical()
        sc.category.category = inputs["label_str"]
        return pb2.Actual(
            actual_label=pb2.ActualLabel(score_categorical=sc),
        )
    elif type == "score_categorical_int" or type == "generative_int":
        sc = pb2.ScoreCategorical()
        sc.score_value.value = inputs["label_int"]
        return pb2.Actual(
            actual_label=pb2.ActualLabel(score_categorical=sc),
        )
    elif type == "score_categorical_float" or type == "generative_float":
        sc = pb2.ScoreCategorical()
        sc.score_value.value = inputs["label_float"]
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
    elif type == "multi_class":
        actual_labels = []
        for class_name, score in inputs["multi_class_actual_scores"].items():
            if score == 1:
                actual_labels.append(class_name)
        mc = pb2.MultiClassActual(
            actual_labels=actual_labels,
        )
        return pb2.Actual(
            actual_label=pb2.ActualLabel(multi_class=mc),
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


def _attach_features_to_prediction(replace: dict = None) -> pb2.Prediction:
    features = {
        "feature_str": pb2.Value(string=STR_VAL),
        "feature_double": pb2.Value(double=FLOAT_VAL),
        "feature_int": pb2.Value(int=INT_VAL),
        "feature_bool": pb2.Value(string=str(BOOL_VAL)),
        "feature_str_lst": pb2.Value(multi_value=pb2.MultiValue(values=STR_LST_VAL)),
    }
    if replace:
        features.update(replace)
    return pb2.Prediction(features=features)


def _attach_llm_field_to_prediction(
    prompt_template=None, prompt_template_version=None, llm_model_name=None, llm_params=None
) -> pb2.Prediction:
    llm_fields = pb2.LLMFields(
        prompt_template=prompt_template or "",
        prompt_template_name=prompt_template_version or "",
        llm_model_name=llm_model_name or "",
        llm_params=convert_dictionary(llm_params),
    )
    return pb2.Prediction(llm_fields=llm_fields)


def _attach_llm_run_metadata_to_prediction(
    prediction=pb2.Prediction,
    llm_run_metadata=LLMRunMetadata,
) -> pb2.Prediction:
    tags = {
        LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME: pb2.Value(
            int=llm_run_metadata.total_token_count
        ),
        LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME: pb2.Value(
            int=llm_run_metadata.prompt_token_count
        ),
        LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME: pb2.Value(
            int=llm_run_metadata.response_token_count
        ),
        LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME: pb2.Value(
            int=llm_run_metadata.response_latency_ms
        ),
    }
    for key, value in prediction.tags.items():
        tags[key] = value
    return pb2.Prediction(tags=tags)


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


def _attach_prompt_and_response_to_prediction(
    input_prompt: Optional[Union[str, Embedding]], input_response: Optional[Union[str, Embedding]]
) -> pb2.Prediction:
    embedding_features = {}
    if input_prompt is not None:
        if isinstance(input_prompt, Embedding):
            prompt_value = pb2.Value(
                embedding=pb2.Embedding(
                    vector=input_prompt.vector,
                    raw_data=pb2.Embedding.RawData(
                        tokenArray=pb2.Embedding.TokenArray(
                            tokens=[input_prompt.data],  # List of a single string
                        )
                    ),
                    link_to_data=StringValue(value=input_prompt.link_to_data),
                )
            )
        elif isinstance(input_prompt, str):
            prompt_value = pb2.Value(
                embedding=pb2.Embedding(
                    raw_data=pb2.Embedding.RawData(
                        tokenArray=pb2.Embedding.TokenArray(
                            tokens=[input_prompt],  # List of a single string
                        )
                    ),
                )
            )
        embedding_features.update(
            {
                "prompt": prompt_value,
            }
        )
    if input_response is not None:
        if isinstance(input_response, Embedding):
            response_value = pb2.Value(
                embedding=pb2.Embedding(
                    vector=input_response.vector,
                    raw_data=pb2.Embedding.RawData(
                        tokenArray=pb2.Embedding.TokenArray(
                            tokens=[input_response.data],  # List of a single string
                        )
                    ),
                    link_to_data=StringValue(value=input_response.link_to_data),
                )
            )
        elif isinstance(input_response, str):
            response_value = pb2.Value(
                embedding=pb2.Embedding(
                    raw_data=pb2.Embedding.RawData(
                        tokenArray=pb2.Embedding.TokenArray(
                            tokens=[input_response],  # List of a single string
                        )
                    ),
                )
            )
        embedding_features.update(
            {
                "response": response_value,
            }
        )

    return pb2.Prediction(features=embedding_features)


def _attach_tags_to_prediction(replace: dict = None) -> pb2.Prediction:
    tags = {
        "tag_str": pb2.Value(string=STR_VAL),
        "tag_double": pb2.Value(double=FLOAT_VAL),
        "tag_int": pb2.Value(int=INT_VAL),
        "tag_bool": pb2.Value(string=str(BOOL_VAL)),
    }
    if replace:
        tags.update(replace)
    return pb2.Prediction(tags=tags)


def _attach_tags_to_actual(replace: dict = None) -> pb2.Actual:
    tags = {
        "tag_str": pb2.Value(string=STR_VAL),
        "tag_double": pb2.Value(double=FLOAT_VAL),
        "tag_int": pb2.Value(int=INT_VAL),
        "tag_bool": pb2.Value(string=str(BOOL_VAL)),
    }
    if replace:
        tags.update(replace)
    return pb2.Actual(tags=tags)


def get_stubbed_client(additional_headers=None):
    c = Client(
        space_key=inputs["space_key"],
        api_key=inputs["api_key"],
        uri="https://localhost:443",
        additional_headers=additional_headers,
    )

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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_bool"],
        actual_label=inputs["label_bool"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_str"],
        actual_label=inputs["label_str"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.NUMERIC,
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.REGRESSION,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_int"],
        actual_label=inputs["label_int"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.NUMERIC,
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.REGRESSION,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_float"],
        actual_label=inputs["label_float"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.SCORE_CATEGORICAL,
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.BINARY_CLASSIFICATION,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_tuple"],
        actual_label=inputs["label_tuple"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.RANKING,
        prediction_id=inputs["prediction_id"],
        prediction_label=pred_label,
        actual_label=act_label,
        features=inputs["features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
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


def test_build_pred_and_actual_label_multi_class_multi_label():
    pred_label = MultiClassPredictionLabel(
        prediction_scores=inputs["multi_class_prediction_scores"],
        threshold_scores=inputs["multi_class_threshold_scores"],
    )
    act_label = MultiClassActualLabel(
        actual_scores=inputs["multi_class_actual_scores"],
    )
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.MULTI_CLASS,
        prediction_id=inputs["prediction_id"],
        prediction_label=pred_label,
        actual_label=act_label,
        features=inputs["features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("multi_class")
    a = _build_basic_actual("multi_class")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Add props to prediction according to this test
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, a=a, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_build_pred_and_actual_label_multi_class_single_label():
    pred_label = MultiClassPredictionLabel(
        prediction_scores=inputs["multi_class_prediction_scores"],
    )
    act_label = MultiClassActualLabel(
        actual_scores=inputs["multi_class_actual_scores"],
    )
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.MULTI_CLASS,
        prediction_id=inputs["prediction_id"],
        prediction_label=pred_label,
        actual_label=act_label,
        features=inputs["features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
    #   Start constructing expected result by building the prediction
    prediction_scores_with_double_value = {}
    for class_name, score in inputs["multi_class_prediction_scores"].items():
        prediction_scores_with_double_value[class_name] = DoubleValue(value=score)
    single_label = pb2.MultiClassPrediction.SingleLabel(
        prediction_scores=prediction_scores_with_double_value,
    )
    p = pb2.Prediction(
        prediction_label=pb2.PredictionLabel(
            multi_class=pb2.MultiClassPrediction(single_label=single_label)
        ),
        model_version=inputs["model_version"],
    )
    a = _build_basic_actual("multi_class")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Add props to prediction according to this test
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, a=a, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_build_wrong_timestamp():
    c = get_stubbed_client()
    wrong_min_time = int(time.time()) - (MAX_PAST_YEARS_FROM_CURRENT_TIME * 365 * 24 * 60 * 60 + 1)
    wrong_max_time = int(time.time()) + (
        MAX_FUTURE_YEARS_FROM_CURRENT_TIME * 365 * 24 * 60 * 60 + 1
    )

    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            prediction_timestamp=wrong_min_time,
            model_type=ModelTypes.NUMERIC,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_float"],
            features=inputs["features"],
            tags=inputs["tags"],
        )
    assert f"prediction_timestamp: {wrong_min_time} is out of range." in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            prediction_timestamp=wrong_max_time,
            model_type=ModelTypes.NUMERIC,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_float"],
            features=inputs["features"],
            tags=inputs["tags"],
        )
    assert f"prediction_timestamp: {wrong_max_time} is out of range." in str(excinfo.value)


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
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.RANKING,
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            actual_label=act_label,
            features=inputs["features"],
            tags=inputs["tags"],
        )
    assert "Rank must be between 1 and 100, inclusive. Found 101" in str(excinfo.value)


def test_ranking_group_id():
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
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.RANKING,
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
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.RANKING,
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
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.RANKING,
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
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.RANKING,
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.OBJECT_DETECTION,
        prediction_id=inputs["prediction_id"],
        prediction_label=pred_label,
        actual_label=act_label,
        features=inputs["features"],
        embedding_features=inputs["object_detection_embedding_feature"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.NUMERIC,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_float"],
        features=inputs["features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.NUMERIC,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_float"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.NUMERIC,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_float"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.NUMERIC,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_float"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric_float")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_build_actual_only_tags_with_prediction_label():
    # This tests that tags are included in the prediction, no actual created
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.NUMERIC,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_float"],
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("numeric_float")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_build_actual_only_tags_with_no_prediction_label():
    # This tests that a label-less actual is created, for latent tags
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.NUMERIC,
        prediction_id=inputs["prediction_id"],
        tags=inputs["tags"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
    #   Start constructing expected result by building the actual
    a = _build_basic_actual()  # label-less
    #   Add props to prediction according to this test
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(a=a, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_reject_not_enough_data():
    c = get_stubbed_client()
    # Test that one of prediction, actual, or feature importance is sent
    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.NUMERIC,
            prediction_id=inputs["prediction_id"],
        )
    assert (
        "must provide at least one of prediction_label, actual_label, tags, or shap_values"
        in str(excinfo.value)
    )


def test_build_prediction_no_tags_no_features():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.NUMERIC,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_float"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
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
            environment=Environments.PRODUCTION,
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
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.NUMERIC,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_float"],
    )

    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
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
            model_type=ModelTypes.NUMERIC,
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
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.OBJECT_DETECTION,
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
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.OBJECT_DETECTION,
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
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.OBJECT_DETECTION,
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
            model_type=ModelTypes.BINARY_CLASSIFICATION,
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            prediction_id="A" * (MAX_PREDICTION_ID_LEN + 1),
            prediction_label=inputs["label_str"],
            actual_label=inputs["label_str"],
            features=inputs["features"],
            embedding_features=inputs["embedding_features"],
            tags=inputs["tags"],
        )
    assert "The string length of prediction_id" in str(excinfo.value)


def test_prediction_id():
    c = get_stubbed_client()

    correct_cases = [
        {
            # test case - None prediction_id, Training
            "prediction_id": None,
            "environment": Environments.TRAINING,
            "prediction_label": inputs["label_str"],
        },
        {
            # test case - None prediction_id, Production, not a delayed record
            "prediction_id": None,
            "environment": Environments.PRODUCTION,
            "prediction_label": inputs["label_str"],
        },
    ]
    for case in correct_cases:
        try:
            _ = c.log(
                model_id=inputs["model_id"],
                model_type=ModelTypes.BINARY_CLASSIFICATION,
                model_version=inputs["model_version"],
                environment=case["environment"],
                prediction_id=case["prediction_id"],
                prediction_label=case["prediction_label"],
                actual_label=inputs["label_str"],
                features=inputs["features"],
                embedding_features=inputs["embedding_features"],
                tags=inputs["tags"],
            )
        except Exception as e:
            msg = (
                f"Logging data without prediction_id raised an exception {e}. "
                + f"prediction_id={case['prediction_id']}, environment={case['environment']}, "
                + f"prediction_label={case['prediction_label']}."
            )
            assert False, msg

    short_prediction_id = "x" * (MIN_PREDICTION_ID_LEN - 1)
    long_prediction_id = "x" * (MAX_PREDICTION_ID_LEN + 1)
    incorrect_cases = [
        {
            # test case - None prediction_id, Production, delayed record
            "prediction_id": None,
            "environment": Environments.PRODUCTION,
            "prediction_label": None,
            "err_msg": "prediction_id value cannot be None for delayed records",
        },
        {
            # test case - Wrong length prediction_id, Training
            "prediction_id": short_prediction_id,
            "environment": Environments.TRAINING,
            "prediction_label": inputs["label_str"],
            "err_msg": f"The string length of prediction_id {short_prediction_id} must be between",
        },
        {
            # test case - Wrong length prediction_id, Production, delayed record
            "prediction_id": short_prediction_id,
            "environment": Environments.PRODUCTION,
            "prediction_label": None,
            "err_msg": f"The string length of prediction_id {short_prediction_id} must be between",
        },
        {
            # test case - Wrong length prediction_id, Production, not a delayed record
            "prediction_id": long_prediction_id,
            "environment": Environments.PRODUCTION,
            "prediction_label": inputs["label_str"],
            "err_msg": f"The string length of prediction_id {long_prediction_id} must be between",
        },
    ]
    for case in incorrect_cases:
        with pytest.raises(ValueError) as exc_info:
            _ = c.log(
                model_id=inputs["model_id"],
                model_type=ModelTypes.BINARY_CLASSIFICATION,
                model_version=inputs["model_version"],
                environment=case["environment"],
                prediction_id=case["prediction_id"],
                prediction_label=case["prediction_label"],
                actual_label=inputs["label_str"],
                features=inputs["features"],
                embedding_features=inputs["embedding_features"],
                tags=inputs["tags"],
            )
        assert case["err_msg"] in str(exc_info.value)


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
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.OBJECT_DETECTION,
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
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.OBJECT_DETECTION,
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
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.OBJECT_DETECTION,
            prediction_id=inputs["prediction_id"],
            prediction_label=pred_label,
            features=inputs["features"],
            embedding_features=inputs["object_detection_embedding_feature"],
            tags=inputs["tags"],
        )
    assert "Bounding box confidence scores must be between 0 and 1, inclusive" in str(excinfo.value)


def test_valid_generative_model():
    c = get_stubbed_client()
    # Test allowed label types
    for label_type in ["str", "bool", "int", "float"]:
        for prompt_opt, response_opt in [
            ("prompt_embedding", "response_embedding"),
            ("prompt_embedding", "response_str"),
            ("prompt_str", "response_embedding"),
            ("prompt_str", "response_str"),
            (None, "response_str"),
            ("prompt_str", None),
        ]:
            prediction_label = inputs[f"label_{label_type}"]
            actual_label = prediction_label
            if prompt_opt is None:
                input_prompt = None
            else:
                input_prompt = inputs[prompt_opt]
            if response_opt is None:
                input_response = None
            else:
                input_response = inputs[response_opt]
            try:
                record = c.log(
                    model_id=inputs["model_id"],
                    model_version=inputs["model_version"],
                    environment=Environments.PRODUCTION,
                    model_type=ModelTypes.GENERATIVE_LLM,
                    prediction_id=inputs["prediction_id"],
                    prediction_label=prediction_label,
                    actual_label=actual_label,
                    features=inputs["features"],
                    embedding_features=inputs["embedding_features"],
                    tags=inputs["tags"],
                    prompt=input_prompt,
                    response=input_response,
                )
            except Exception:
                pytest.fail("Unexpected error!")
            #   Get environment in proto format
            ep = _get_proto_environment_params(Environments.PRODUCTION)
            #   Start constructing expected result by building the prediction
            p = _build_basic_prediction(f"generative_{label_type}")
            a = _build_basic_actual(f"generative_{label_type}")
            #   Add props to prediction according to this test
            p.MergeFrom(_attach_features_to_prediction())
            p.MergeFrom(_attach_embedding_features_to_prediction())
            p.MergeFrom(
                _attach_prompt_and_response_to_prediction(
                    input_prompt=input_prompt, input_response=input_response
                )
            )
            p.MergeFrom(_attach_tags_to_prediction())
            #   Add props to actual according to this test
            a.MergeFrom(_attach_tags_to_actual())
            #   Build expected record using built prediction
            expected_record = _build_expected_record(
                p=p, a=a, ep=ep, is_generative_llm_record=BoolValue(value=True)
            )
            #   Check result is as expected
            assert record == expected_record


def test_generative_if_no_prediction_or_actual_label():
    @dataclass
    class TestConfig:
        name: str
        prediction_label: str
        actual_label: str
        expected_prediction_label: str
        expected_actual_label: str

    tests = [
        TestConfig(
            name="no_prediction_label_or_actual_label",  # include default prediction_label
            prediction_label=None,
            actual_label=None,
            expected_prediction_label="1",
            expected_actual_label=None,
        ),
        # Testing the case that allows us to still support latent actual
        TestConfig(
            name="no_prediction_label",
            prediction_label=None,
            actual_label="1",
            expected_prediction_label=None,
            expected_actual_label="1",
        ),
        TestConfig(
            name="no_actual_label",
            prediction_label="1",
            actual_label=None,
            expected_prediction_label="1",
            expected_actual_label=None,
        ),
        TestConfig(
            name="include_prediction_and_actual",
            prediction_label="1",
            actual_label="1",
            expected_prediction_label="1",
            expected_actual_label="1",
        ),
    ]

    for test in tests:
        c = get_stubbed_client()
        # Test allowed label types
        input_prompt = inputs["prompt_embedding"]
        input_response = inputs["response_embedding"]
        try:
            record = c.log(
                model_id=inputs["model_id"],
                model_version=inputs["model_version"],
                environment=Environments.PRODUCTION,
                model_type=ModelTypes.GENERATIVE_LLM,
                prediction_id=inputs["prediction_id"],
                features=inputs["features"],
                embedding_features=inputs["embedding_features"],
                tags=inputs["tags"],
                prediction_label=test.prediction_label,
                actual_label=test.actual_label,
                prompt=input_prompt,
                response=input_response,
            )
        except Exception:
            pytest.fail(f"Unexpected error on test={test.name}")
        #   Get environment in proto format
        ep = _get_proto_environment_params(Environments.PRODUCTION)
        p = None
        if test.expected_prediction_label is not None:
            # Start constructing expected result by building the prediction
            # This prediction was not passed to the log call, but should be
            # created by default for GENERATIVE models
            sc = pb2.ScoreCategorical()
            sc.category.category = test.expected_prediction_label
            p = pb2.Prediction(
                prediction_label=pb2.PredictionLabel(score_categorical=sc),
                model_version=inputs["model_version"],
            )
            #   Add props to prediction according to this test
            p.MergeFrom(_attach_features_to_prediction())
            p.MergeFrom(_attach_embedding_features_to_prediction())
            p.MergeFrom(
                _attach_prompt_and_response_to_prediction(
                    input_prompt=input_prompt, input_response=input_response
                )
            )
            p.MergeFrom(_attach_tags_to_prediction())
        a = None
        if test.expected_actual_label is not None:
            #   Start constructing expected result by building the actual
            sc = pb2.ScoreCategorical()
            sc.category.category = test.expected_actual_label
            a = pb2.Actual(
                actual_label=pb2.ActualLabel(score_categorical=sc),
            )
            #   Add props to prediction according to this test
            a.MergeFrom(_attach_tags_to_actual())

        #   Build expected record
        expected_record = _build_expected_record(
            p=p, a=a, ep=ep, is_generative_llm_record=BoolValue(value=True)
        )
        #   Check result is as expected
        assert record == expected_record, test.name


def test_invalid_prompt_response_input_generative_model():
    c = get_stubbed_client()
    # Test that GENERATIVE_LLM models cannot contain embedding named prompt or response
    with pytest.raises(KeyError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.GENERATIVE_LLM,
            prediction_id=inputs["prediction_id"],
            prediction_label=1,
            features=inputs["features"],
            tags=inputs["tags"],
            embedding_features={"prompt": inputs["prompt_embedding"]},
            prompt=inputs["prompt_embedding"],
            response=inputs["response_embedding"],
        )
    assert (
        "embedding features cannot use the reserved feature names ('prompt', 'response') "
        "for GENERATIVE_LLM models" in str(excinfo.value)
    )
    # Test that prompt must be str or Embedding type for GENERATIVE_LLM models
    with pytest.raises(TypeError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.GENERATIVE_LLM,
            prediction_id=inputs["prediction_id"],
            prediction_label=1,
            features=inputs["features"],
            tags=inputs["tags"],
            prompt=2,
            response=inputs["response_embedding"],
        )
    assert "prompt must be of type str or Embedding" in str(excinfo.value)
    # Test that response must be str or Embedding type for GENERATIVE_LLM models
    with pytest.raises(TypeError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.GENERATIVE_LLM,
            prediction_id=inputs["prediction_id"],
            prediction_label=1,
            features=inputs["features"],
            tags=inputs["tags"],
            prompt=inputs["prompt_embedding"],
            response=2,
        )
    assert "response must be of type str or Embedding" in str(excinfo.value)


def test_invalid_prompt_response_input_for_model_type():
    c = get_stubbed_client()

    # Test that 'prompt' must be None for models other than GENERATIVE_LLM
    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.SCORE_CATEGORICAL,
            prediction_id=inputs["prediction_id"],
            prediction_label=1,
            features=inputs["features"],
            tags=inputs["tags"],
            prompt=inputs["prompt_embedding"],
        )
    assert (
        "The fields 'prompt' and 'response' must be None for model types other than GENERATIVE_LLM"
        in str(excinfo.value)
    )
    # Test that 'response' must be None for models other than GENERATIVE_LLM
    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.SCORE_CATEGORICAL,
            prediction_id=inputs["prediction_id"],
            prediction_label=1,
            features=inputs["features"],
            tags=inputs["tags"],
            response=inputs["response_embedding"],
        )
    assert (
        "The fields 'prompt' and 'response' must be None for model types other than GENERATIVE_LLM"
        in str(excinfo.value)
    )


def test_invalid_generative_prompt_template_and_llm_config_types():
    c = get_stubbed_client()
    # Test that 'llm_model_name' must be a string with limited amount of characters
    with pytest.raises(err.InvalidValueType) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.GENERATIVE_LLM,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_int"],
            features=inputs["features"],
            tags=inputs["tags"],
            prompt=inputs["prompt_embedding"],
            response=inputs["response_embedding"],
            llm_model_name=0,
        )
    assert "llm_model_name" in str(excinfo.value)
    assert "str" in str(excinfo.value)
    # Test that 'prompt_template' must be a string with limited amount of characters
    with pytest.raises(err.InvalidValueType) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.GENERATIVE_LLM,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_int"],
            features=inputs["features"],
            tags=inputs["tags"],
            prompt=inputs["prompt_embedding"],
            response=inputs["response_embedding"],
            prompt_template=0,
        )
    assert "prompt_template" in str(excinfo.value)
    assert "str" in str(excinfo.value)
    # Test that 'prompt_template_version' must be a string with limited amount of characters
    with pytest.raises(err.InvalidValueType) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.GENERATIVE_LLM,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_int"],
            features=inputs["features"],
            tags=inputs["tags"],
            prompt=inputs["prompt_embedding"],
            response=inputs["response_embedding"],
            prompt_template_version=0,
        )
    assert "prompt_template_version" in str(excinfo.value)
    assert "str" in str(excinfo.value)
    # Test that 'llm_params' must be a string with limited amount of characters
    with pytest.raises(err.InvalidValueType) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.GENERATIVE_LLM,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_int"],
            features=inputs["features"],
            tags=inputs["tags"],
            prompt=inputs["prompt_embedding"],
            response=inputs["response_embedding"],
            llm_params=0,
        )
    assert "llm_params" in str(excinfo.value)
    assert "dict" in str(excinfo.value)


def test_valid_generative_prompt_template_and_llm_config():
    c = get_stubbed_client()
    llm_model_names = [None, "gpt-4"]
    prompt_templates = [None, "This is a test template with context {{context}}"]
    prompt_template_versions = [None, "Template A"]
    llm_params_list = [None, {"temperature": 0.9, "presence_pnlty": 0.34, "stop": [".", "?", "!"]}]
    # Test allowed label types
    combinations = list(
        itertools.product(
            *[llm_model_names, prompt_templates, prompt_template_versions, llm_params_list]
        )
    )
    label_type = "int"
    prediction_label = inputs[f"label_{label_type}"]
    actual_label = prediction_label
    input_prompt = inputs["prompt_embedding"]
    input_response = inputs["response_embedding"]
    for comb in combinations:
        llm_model_name, prompt_template, prompt_template_version, llm_params = comb
        record = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.GENERATIVE_LLM,
            prediction_id=inputs["prediction_id"],
            prediction_label=prediction_label,
            actual_label=actual_label,
            features=inputs["features"],
            embedding_features=inputs["embedding_features"],
            tags=inputs["tags"],
            prompt=input_prompt,
            response=input_response,
            llm_model_name=llm_model_name,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            llm_params=llm_params,
        )
        #   Get environment in proto format
        ep = _get_proto_environment_params(Environments.PRODUCTION)
        #   Start constructing expected result by building the prediction
        p = _build_basic_prediction(f"generative_{label_type}")
        a = _build_basic_actual(f"generative_{label_type}")
        #   Add props to prediction according to this test
        p.MergeFrom(_attach_features_to_prediction())
        p.MergeFrom(_attach_embedding_features_to_prediction())
        p.MergeFrom(
            _attach_prompt_and_response_to_prediction(
                input_prompt=input_prompt, input_response=input_response
            )
        )
        p.MergeFrom(_attach_tags_to_prediction())
        if llm_model_name or prompt_template or prompt_template_version or llm_params:
            p.MergeFrom(
                _attach_llm_field_to_prediction(
                    prompt_template=prompt_template,
                    prompt_template_version=prompt_template_version,
                    llm_model_name=llm_model_name,
                    llm_params=llm_params,
                )
            )
        #   Add props to actual according to this test
        a.MergeFrom(_attach_tags_to_actual())
        #   Build expected record using built prediction
        expected_record = _build_expected_record(
            p=p, a=a, ep=ep, is_generative_llm_record=BoolValue(value=True)
        )
        #   Check result is as expected
        assert record == expected_record


def test_invalid_llm_run_metadata():
    c = get_stubbed_client()
    with pytest.raises(err.InvalidValueType) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.GENERATIVE_LLM,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_int"],
            features=inputs["features"],
            tags=inputs["tags"],
            prompt=inputs["prompt_embedding"],
            response=inputs["response_embedding"],
            llm_run_metadata=LLMRunMetadata(
                total_token_count="string is invalid",
                prompt_token_count=200,
                response_token_count=100,
                response_latency_ms=1000,
            ),
        )
    assert "total_token_count" in str(excinfo.value)
    with pytest.raises(err.InvalidValueType) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.GENERATIVE_LLM,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_int"],
            features=inputs["features"],
            tags=inputs["tags"],
            prompt=inputs["prompt_embedding"],
            response=inputs["response_embedding"],
            llm_run_metadata=LLMRunMetadata(
                total_token_count=300,
                prompt_token_count="string is invalid",
                response_token_count=100,
                response_latency_ms=1000,
            ),
        )
    assert "prompt_token_count" in str(excinfo.value)
    with pytest.raises(err.InvalidValueType) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.GENERATIVE_LLM,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_int"],
            features=inputs["features"],
            tags=inputs["tags"],
            prompt=inputs["prompt_embedding"],
            response=inputs["response_embedding"],
            llm_run_metadata=LLMRunMetadata(
                total_token_count=300,
                prompt_token_count=200,
                response_token_count="string is invalid",
                response_latency_ms=1000,
            ),
        )
    assert "response_token_count" in str(excinfo.value)
    with pytest.raises(err.InvalidValueType) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.GENERATIVE_LLM,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_int"],
            features=inputs["features"],
            tags=inputs["tags"],
            prompt=inputs["prompt_embedding"],
            response=inputs["response_embedding"],
            llm_run_metadata=LLMRunMetadata(
                total_token_count=300,
                prompt_token_count=200,
                response_token_count=100,
                response_latency_ms="string is invalid",
            ),
        )
    assert "response_latency_ms" in str(excinfo.value)


def test_valid_llm_run_metadata():
    c = get_stubbed_client()
    label_type = "int"
    prediction_label = inputs[f"label_{label_type}"]
    actual_label = prediction_label
    llm_model_name = "gpt-4"
    prompt_template = "This is a test template with context {{context}}"
    prompt_template_version = "Template A"
    llm_params = {"temperature": 0.9, "presence_pnlty": 0.34, "stop": [".", "?", "!"]}
    llm_run_metadata = LLMRunMetadata(
        total_token_count=300,
        prompt_token_count=200,
        response_token_count=100,
        response_latency_ms=1000,
    )
    input_prompt = inputs["prompt_embedding"]
    input_response = inputs["response_embedding"]
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.GENERATIVE_LLM,
        prediction_id=inputs["prediction_id"],
        prediction_label=prediction_label,
        actual_label=actual_label,
        features=inputs["features"],
        embedding_features=inputs["embedding_features"],
        tags=inputs["tags"],
        prompt=input_prompt,
        response=input_response,
        llm_model_name=llm_model_name,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        llm_params=llm_params,
        llm_run_metadata=llm_run_metadata,
    )
    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction(f"generative_{label_type}")
    a = _build_basic_actual(f"generative_{label_type}")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(
        _attach_prompt_and_response_to_prediction(
            input_prompt=input_prompt, input_response=input_response
        )
    )
    p.MergeFrom(_attach_tags_to_prediction())
    p.MergeFrom(
        _attach_llm_field_to_prediction(
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            llm_model_name=llm_model_name,
            llm_params=llm_params,
        )
    )
    p.MergeFrom(
        _attach_llm_run_metadata_to_prediction(
            prediction=p,
            llm_run_metadata=llm_run_metadata,
        )
    )
    #   Add props to actual according to this test
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(
        p=p, a=a, ep=ep, is_generative_llm_record=BoolValue(value=True)
    )
    #   Check result is as expected
    assert record == expected_record


def test_reserved_tags():
    c = get_stubbed_client()
    wrong_tags = [reserved_tag_col for reserved_tag_col in RESERVED_TAG_COLS]
    for wrong_tag in wrong_tags:
        # test case - too long tag value
        with pytest.raises(KeyError) as excinfo:
            _ = c.log(
                model_id=inputs["model_id"],
                model_type=ModelTypes.BINARY_CLASSIFICATION,
                model_version=inputs["model_version"],
                environment=Environments.PRODUCTION,
                prediction_id=inputs["prediction_id"],
                prediction_label=inputs["label_str"],
                actual_label=inputs["label_str"],
                features=inputs["features"],
                tags={wrong_tag: 100},
            )
        assert "The following tag names are not allowed as they are reserved" in str(
            excinfo
        ) and wrong_tag in str(excinfo)


def test_invalid_tags():
    c = get_stubbed_client()
    wrong_tags = {
        "tag_str_incorrect": "a" * (MAX_TAG_LENGTH + 1),
    }

    # test case - too long tag value
    with pytest.raises(ValueError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_type=ModelTypes.BINARY_CLASSIFICATION,
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_str"],
            actual_label=inputs["label_str"],
            features=inputs["features"],
            tags=wrong_tags,
        )
    assert (
        f"The number of characters for each tag must be less than or equal to {MAX_TAG_LENGTH}."
        in str(excinfo.value)
    )


def test_instantiating_client_duplicated_header():
    with pytest.raises(InvalidAdditionalHeaders) as excinfo:
        _ = get_stubbed_client({"authorization": "FAKE_VALUE"})
    assert (
        "Found invalid additional header, cannot use reserved headers named: authorization."
        in str(excinfo.value)
    )


def test_instantiating_client_additional_header():
    c = get_stubbed_client({"JWT": "FAKE_VALUE"})

    expected = {
        "authorization": inputs["api_key"],
        "Grpc-Metadata-space": inputs["space_key"],
        "Grpc-Metadata-space_id": None,
        "Grpc-Metadata-sdk-language": "python",
        "Grpc-Metadata-language-version": get_python_version(),
        "Grpc-Metadata-sdk-version": arize_version,
        "JWT": "FAKE_VALUE",
    }
    assert c._headers == expected


def test_invalid_client_auth_passed_vars():
    with pytest.raises(err.AuthError) as excinfo:
        _ = Client()
    assert excinfo.value.__str__() == err.AuthError().error_message()
    assert "Missing: ['api_key', 'space_id']" in str(excinfo.value)

    with pytest.raises(err.AuthError) as excinfo:
        _ = Client(space_key=inputs["space_key"])
    assert excinfo.value.__str__() == err.AuthError(space_key=inputs["space_key"]).error_message()
    assert "Missing: ['api_key']" in str(excinfo.value)

    with pytest.raises(err.AuthError) as excinfo:
        _ = Client(api_key=inputs["api_key"])
    assert excinfo.value.__str__() == err.AuthError(api_key=inputs["api_key"]).error_message()
    assert "Missing: ['space_id']" in str(excinfo.value)

    # incorrect type
    with pytest.raises(err.InvalidTypeAuthKey) as excinfo:
        _ = Client(api_key=123, space_key="space_key")
    assert (
        excinfo.value.__str__()
        == err.InvalidTypeAuthKey(api_key=123, space_key="space_key").error_message()
    )
    assert "api_key as int" in str(excinfo.value)

    with pytest.raises(err.InvalidTypeAuthKey) as excinfo:
        api_key = "api_key"
        space_key = (
            "space_key",
        )  # This comma is intentional to make space_key an accidental tuple
        _ = Client(api_key=api_key, space_key=space_key)
    assert (
        excinfo.value.__str__()
        == err.InvalidTypeAuthKey(api_key=api_key, space_key=space_key).error_message()
    )
    assert "space_key as tuple" in str(excinfo.value)

    # acceptable input
    try:
        _ = Client(space_key=inputs["space_key"], api_key=inputs["api_key"])
    except Exception:
        pytest.fail("Unexpected error!")

    try:
        _ = Client(space_id=inputs["space_id"], api_key=inputs["api_key"])
    except Exception:
        pytest.fail("Unexpected error!")


def test_invalid_client_auth_environment_vars(monkeypatch):
    with pytest.raises(err.AuthError) as excinfo:
        _ = Client()
    assert excinfo.value.__str__() == err.AuthError().error_message()
    assert "Missing: ['api_key', 'space_id']" in str(excinfo.value)

    monkeypatch.setenv(SPACE_KEY_ENVVAR_NAME, inputs["space_key"])
    with pytest.raises(err.AuthError) as excinfo:
        c = Client()
        assert c._space_key == inputs["space_key"]
    assert excinfo.value.__str__() == err.AuthError(space_key=inputs["space_key"]).error_message()
    assert "Missing: ['api_key']" in str(excinfo.value)

    monkeypatch.delenv(SPACE_KEY_ENVVAR_NAME)
    monkeypatch.setenv(API_KEY_ENVVAR_NAME, inputs["api_key"])
    with pytest.raises(err.AuthError) as excinfo:
        c = Client()
        assert c._api_key == inputs["api_key"]
    assert excinfo.value.__str__() == err.AuthError(api_key=inputs["api_key"]).error_message()
    assert "Missing: ['space_id']" in str(excinfo.value)

    # acceptable input
    monkeypatch.setenv(SPACE_KEY_ENVVAR_NAME, inputs["space_key"])
    try:
        c = Client()
    except Exception:
        pytest.fail("Unexpected error!")
    assert c._space_key == inputs["space_key"]
    assert c._api_key == inputs["api_key"]
    assert c._space_id is None

    monkeypatch.delenv(SPACE_KEY_ENVVAR_NAME)
    monkeypatch.setenv(SPACE_ID_ENVVAR_NAME, inputs["space_id"])
    try:
        c = Client()
    except Exception:
        pytest.fail("Unexpected error!")
    assert c._space_key is None
    assert c._api_key == inputs["api_key"]
    assert c._space_id == inputs["space_id"]


def test_invalid_number_of_embeddings():
    c = get_stubbed_client()
    # Test failure
    N = MAX_NUMBER_OF_EMBEDDINGS + 1
    embedding_features = {
        f"embedding_feat_{i:02d}": inputs["embedding_features"]["image_embedding"] for i in range(N)
    }
    with pytest.raises(InvalidNumberOfEmbeddings) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.SCORE_CATEGORICAL,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_bool"],
            actual_label=inputs["label_bool"],
            features=inputs["features"],
            embedding_features=embedding_features,
            tags=inputs["tags"],
        )
    err_msg = (
        f"The schema contains {N} different embeddings when a maximum of "
        f"{MAX_NUMBER_OF_EMBEDDINGS} is allowed."
    )
    assert err_msg in str(excinfo.value)

    # Test success
    N = MAX_NUMBER_OF_EMBEDDINGS
    embedding_features = {
        f"embedding_feat_{i:02d}": inputs["embedding_features"]["image_embedding"] for i in range(N)
    }
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_bool"],
        actual_label=inputs["label_bool"],
        features=inputs["features"],
        embedding_features=embedding_features,
        tags=inputs["tags"],
    )
    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("score_categorical_bool")
    a = _build_basic_actual("score_categorical_bool")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction())
    emb_feats = {}
    for emb_feat_name, emb_feat_value in embedding_features.items():
        emb_feats[emb_feat_name] = pb2.Value(
            embedding=pb2.Embedding(
                vector=emb_feat_value.vector,
                link_to_data=StringValue(value=emb_feat_value.link_to_data),
            )  # type: ignore
        )  # type: ignore
    p.MergeFrom(pb2.Prediction(features=emb_feats))
    p.MergeFrom(_attach_tags_to_prediction())
    #   Add props to prediction according to this test
    a.MergeFrom(_attach_tags_to_actual())
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, a=a, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_casting_success():
    c = get_stubbed_client()
    record = c.log(
        model_id=inputs["model_id"],
        model_version=inputs["model_version"],
        environment=Environments.PRODUCTION,
        model_type=ModelTypes.SCORE_CATEGORICAL,
        prediction_id=inputs["prediction_id"],
        prediction_label=inputs["label_bool"],
        actual_label=inputs["label_bool"],
        features={
            "feature_str": STR_VAL,
            "feature_double": FLOAT_VAL,
            "feature_int": TypedValue(value=INT_VAL, type=ArizeTypes.FLOAT),  # cast int to float
            "feature_bool": BOOL_VAL,
            "feature_None": None,
            "feature_str_lst": STR_LST_VAL,
        },
        tags={
            "tag_str": STR_VAL,
            "tag_double": FLOAT_VAL,
            "tag_int": INT_VAL,
            "tag_bool": TypedValue(value=BOOL_VAL, type=ArizeTypes.STR),  # cast bool to string
            "tag_bool2": TypedValue(value=BOOL_VAL, type=ArizeTypes.INT),  # cast bool to int
            "tag_None": None,
        },
        embedding_features=inputs["embedding_features"],
    )
    f = {
        "feature_int": pb2.Value(double=float(INT_VAL)),
    }
    t = {
        "tag_bool": pb2.Value(string=str(BOOL_VAL)),
        "tag_bool2": pb2.Value(int=int(BOOL_VAL)),
    }
    #   Get environment in proto format
    ep = _get_proto_environment_params(Environments.PRODUCTION)
    #   Start constructing expected result by building the prediction
    p = _build_basic_prediction("score_categorical_bool")
    a = _build_basic_actual("score_categorical_bool")
    #   Add props to prediction according to this test
    p.MergeFrom(_attach_features_to_prediction(replace=f))
    p.MergeFrom(_attach_embedding_features_to_prediction())
    p.MergeFrom(_attach_tags_to_prediction(replace=t))
    #   Add props to prediction according to this test
    a.MergeFrom(_attach_tags_to_actual(replace=t))
    #   Build expected record using built prediction
    expected_record = _build_expected_record(p=p, a=a, ep=ep)
    #   Check result is as expected
    assert record == expected_record


def test_casting_fail():
    c = get_stubbed_client()
    with pytest.raises(CastingError) as excinfo:
        _ = c.log(
            model_id=inputs["model_id"],
            model_version=inputs["model_version"],
            environment=Environments.PRODUCTION,
            model_type=ModelTypes.SCORE_CATEGORICAL,
            prediction_id=inputs["prediction_id"],
            prediction_label=inputs["label_bool"],
            actual_label=inputs["label_bool"],
            features={
                "feature_str": STR_VAL,
                "feature_double": TypedValue(
                    value=FLOAT_VAL, type=ArizeTypes.INT
                ),  # cast float to int - should fail
                "feature_int": INT_VAL,
                "feature_bool": BOOL_VAL,
                "feature_None": None,
                "feature_str_lst": STR_LST_VAL,
            },
            tags={
                "tag_str": STR_VAL,
                "tag_double": FLOAT_VAL,
                "tag_int": INT_VAL,
                "tag_bool": TypedValue(value=BOOL_VAL, type=ArizeTypes.STR),  # cast bool to string
                "tag_bool2": TypedValue(value=BOOL_VAL, type=ArizeTypes.INT),  # cast bool to int
                "tag_None": None,
            },
            embedding_features=inputs["embedding_features"],
        )
    assert excinfo.value.typed_value.value == FLOAT_VAL
    assert excinfo.value.typed_value.type == ArizeTypes.INT
    assert excinfo.value.error_message() == (
        "Failed to cast value 20.2 of type <class 'float'> "
        "to type ArizeTypes.INT. Error: Cannot convert float with "
        "non-zero fractional part to int."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
