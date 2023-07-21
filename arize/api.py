# type: ignore[pb2]
import concurrent.futures as cf
import os
import time
from typing import Dict, Optional, Tuple, Union

from arize.pandas.validation.errors import InvalidAdditionalHeaders
from arize.utils.constants import (
    API_KEY_ENVVAR_NAME,
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
    MAX_PREDICTION_ID_LEN,
    MAX_TAG_LENGTH,
    MIN_PREDICTION_ID_LEN,
    SPACE_KEY_ENVVAR_NAME,
)
from arize.utils.logging import logger
from google.protobuf.json_format import MessageToDict
from google.protobuf.wrappers_pb2 import BoolValue, DoubleValue
from requests_futures.sessions import FuturesSession

from . import public_pb2 as pb2
from .__init__ import __version__
from .bounded_executor import BoundedExecutor
from .utils.errors import AuthError
from .utils.types import (
    CATEGORICAL_MODEL_TYPES,
    NUMERIC_MODEL_TYPES,
    Embedding,
    Environments,
    ModelTypes,
    ObjectDetectionLabel,
    RankingActualLabel,
    RankingPredictionLabel,
)
from .utils.utils import (
    convert_dictionary,
    convert_element,
    get_python_version,
    get_timestamp,
    is_timestamp_in_range,
)

PredictionLabelTypes = Union[
    str,
    bool,
    int,
    float,
    Tuple[str, float],
    ObjectDetectionLabel,
    RankingPredictionLabel,
]
ActualLabelTypes = Union[
    str,
    bool,
    int,
    float,
    Tuple[str, float],
    ObjectDetectionLabel,
    RankingActualLabel,
]


class Client:
    """
    Arize API Client to log model predictions and actuals to the Arize AI platform
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        space_key: Optional[str] = None,
        uri: Optional[str] = "https://api.arize.com/v1",
        max_workers: Optional[int] = 8,
        max_queue_bound: Optional[int] = 5000,
        timeout: Optional[int] = 200,
        additional_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initializes the Arize Client

        Arguments:
        ----------
            api_key (str): Arize provided API key associated with your account. Located on the
                data upload page.
            space_key (str): Arize provided identifier to connect records to spaces. Located on
                the data upload page.
            uri (str, optional): URI to send your records to Arize AI. Defaults to
                "https://api.arize.com/v1".
            max_workers (int, optional): maximum number of concurrent requests to Arize. Defaults
                to 8.
            max_queue_bound (int, optional): maximum number of concurrent future objects
                generated for publishing to Arize. Defaults to 5000.
            timeout (int, optional): How long to wait for the server to send data before giving
                up. Defaults to 200.
            additional_headers (Dict[str, str], optional): Dictionary of additional headers to
                append to request
        """

        api_key = api_key or os.getenv(API_KEY_ENVVAR_NAME)
        space_key = space_key or os.getenv(SPACE_KEY_ENVVAR_NAME)
        if api_key is None or space_key is None:
            raise AuthError(api_key, space_key)
        self._api_key = api_key
        self._space_key = space_key
        self._uri = f"{uri}/log"
        self._timeout = timeout
        self._session = FuturesSession(executor=BoundedExecutor(max_queue_bound, max_workers))
        # Grpc-Metadata prefix is required to pass non-standard md through via grpc-gateway
        self._headers = {
            "authorization": api_key,
            "Grpc-Metadata-space": space_key,
            "Grpc-Metadata-sdk-language": "python",
            "Grpc-Metadata-language-version": get_python_version(),
            "Grpc-Metadata-sdk-version": __version__,
        }
        if additional_headers is not None:
            if conflicting_keys := self._headers.keys() & additional_headers.keys():
                raise InvalidAdditionalHeaders(conflicting_keys)
            self._headers.update(additional_headers)

    def log(
        self,
        model_id: str,
        model_type: ModelTypes,
        environment: Environments,
        model_version: Optional[str] = None,
        prediction_id: Optional[Union[str, int, float]] = None,
        prediction_timestamp: Optional[int] = None,
        prediction_label: Optional[PredictionLabelTypes] = None,
        actual_label: Optional[ActualLabelTypes] = None,
        features: Optional[Dict[str, Union[str, bool, float, int]]] = None,
        embedding_features: Optional[Dict[str, Embedding]] = None,
        shap_values: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, Union[str, bool, float, int]]] = None,
        batch_id: Optional[str] = None,
        prompt: Optional[Embedding] = None,
        response: Optional[Embedding] = None,
    ) -> cf.Future:
        """
        Logs a record to Arize via a POST request.

        Arguments:
        ----------
            model_id (str): A unique name to identify your model in the Arize platform.
            model_type (ModelTypes): Declare your model type. Can check the supported model types
                running `ModelTypes.list_types()`
            environment (Environments): The environment that this dataframe is for (Production,
                Training, Validation).
            model_version (str, optional): Used to group together a subset of predictions
                and actuals for a given model_id to track and compare changes. Defaults to None.
            prediction_id (str, int, or float, optional): A unique string to identify a
                prediction event. This value is used to match a prediction to delayed actuals in Arize. If
                prediction id is not provided, Arize will, when possible, create a random prediction
                id on the server side.
            prediction_timestamp (int, optional): Unix timestamp in seconds. If None, prediction
                uses current timestamp. Defaults to None.
            prediction_label (bool, int, float, str, Tuple(str, float), ObjectDetectionLabel or
                RankingPredictionLabel; optional): The predicted value for a given model input. Defaults to
                None.
            actual_label (bool, int, float, str, Tuple[str, float], ObjectDetectionLabel or
                RankingActualLabel; optional): The ground truth value for a given model input. This will be
                be matched to the prediction with the same prediction_id as the one in this call. Defaults
                to None.
            features (Dict[str, Union[str, bool, float, int]], optional): Dictionary containing
                human readable and debuggable model features. Defaults to None.
            embedding_features (Dict[str, Embedding], optional): Dictionary containing model
                embedding features. Keys must be strings. Values must be of type Embedding. Defaults to
                None.
            shap_values (Dict[str, float], optional): Dictionary containing human readable and
                debuggable model features keys, along with SHAP feature importance values. Defaults to None.
            tags (Dict[str, Union[str, bool, float, int]], optional): Dictionary containing human
                readable and debuggable model tags. Defaults to None.
            batch_id (str, optional): Used to distinguish different batch of data under the same
                model_id and model_version. Required for VALIDATION environments. Defaults to None.
            prompt (Embedding, optional): Embedding object containing the embedding vector (required) and raw
                text (optional, but recommended) for the input text on which your GENERATIVE_LLM model acts
                on. Required for GENERATIVE_LLM models. Defaults to None.
            response (Embedding, optional): Embedding object containing the embedding vector (required) and
                raw text (optional, but recommended) for the text GENERATIVE_LLM model generates.
                Required for GENERATIVE_LLM models. Defaults to None.

        Returns:
        --------
            `concurrent.futures.Future` object
        """
        # Validate model_id
        if not isinstance(model_id, str):
            raise TypeError(f"model_id {model_id} is type {type(model_id)}, but must be a str")
        # Validate model_type
        if not isinstance(model_type, ModelTypes):
            raise TypeError(
                f"model_type {model_type} is type {type(model_type)}, but must be of "
                f"arize.utils.ModelTypes"
            )
        # Validate environment
        if not isinstance(environment, Environments):
            raise TypeError(
                f"environment {environment} is type {type(environment)}, but must be of "
                f"arize.utils.Environments"
            )
        # Validate batch_id
        if environment == Environments.VALIDATION:
            if batch_id is None or not isinstance(batch_id, str) or len(batch_id.strip()) == 0:
                raise ValueError(
                    "Batch ID must be a nonempty string if logging to validation environment."
                )

        # Convert & Validate prediction_id
        prediction_id = _validate_and_convert_prediction_id(
            prediction_id, environment, prediction_label, actual_label, shap_values
        )
        # Validate feature types
        if features:
            for feat_name, feat_value in features.items():
                _validate_mapping_key(feat_name)
                val = convert_element(feat_value)
                if val is not None and not isinstance(val, (str, bool, float, int)):
                    raise TypeError(
                        f"feature {feat_name} with value {feat_value} is type {type(feat_value)}, "
                        f"but expected one of: str, bool, float, int"
                    )

        # Validate embedding_features type
        if embedding_features:
            if model_type == ModelTypes.OBJECT_DETECTION:
                # Check that there is only 1 embedding feature for OD model types
                if len(embedding_features.keys()) > 1:
                    raise ValueError("Object Detection models only support one embedding feature")
            if model_type == ModelTypes.GENERATIVE_LLM:
                # Check reserved keys are not used
                reserved_emb_feat_names = {"prompt", "response"}
                if reserved_emb_feat_names & embedding_features.keys():
                    raise KeyError(
                        "embedding features cannot use the reserved feature names ('prompt', 'response') "
                        "for GENERATIVE_LLM models"
                    )
            for emb_name, emb_obj in embedding_features.items():
                _validate_mapping_key(emb_name)
                # Must verify embedding type
                if not isinstance(emb_obj, Embedding):
                    raise TypeError(f'Embedding feature "{emb_name}" must be of type Embedding')
                emb_obj.validate(emb_name)

        # Validate tag types
        if tags:
            for tag_name, tag_value in tags.items():
                _validate_mapping_key(tag_name)
                val = convert_element(tag_value)
                if val is not None and not isinstance(val, (str, bool, float, int)):
                    raise TypeError(
                        f"tag {tag_name} with value {tag_value} is type {type(tag_value)}, "
                        f"but expected one of: str, bool, float, int"
                    )
                if isinstance(tag_name, str) and tag_name.endswith("_shap"):
                    raise ValueError(f"tag {tag_name} must not be named with a `_shap` suffix")
                if len(str(val)) > MAX_TAG_LENGTH:
                    raise ValueError(
                        f"The number of characters for each tag must be less than or equal to "
                        f"{MAX_TAG_LENGTH}. The tag {tag_name} with value {tag_value} has "
                        f"{len(str(val))} characters."
                    )

        # Check the timestamp present on the event
        if prediction_timestamp is not None:
            if not isinstance(prediction_timestamp, int):
                raise TypeError(
                    f"prediction_timestamp {prediction_timestamp} is type "
                    f"{type(prediction_timestamp)} but expected int"
                )
            # Send warning if prediction is sent with future timestamp
            now = int(time.time())
            if prediction_timestamp > now:
                logger.warning(
                    "Caution when sending a prediction with future timestamp."
                    "Arize only stores 2 years worth of data. For example, if you sent a prediction "
                    "to Arize from 1.5 years ago, and now send a prediction with timestamp of a year in "
                    "the future, the oldest 0.5 years will be dropped to maintain the 2 years worth of data "
                    "requirement."
                )
            if not is_timestamp_in_range(now, prediction_timestamp):
                raise ValueError(
                    f"prediction_timestamp: {prediction_timestamp} is out of range."
                    f"Prediction timestamps must be within {MAX_FUTURE_YEARS_FROM_CURRENT_TIME} year in the "
                    f"future and {MAX_PAST_YEARS_FROM_CURRENT_TIME} years in the past from the current time."
                )

        # Validate GENERATIVE_LLM models requirements
        if model_type == ModelTypes.GENERATIVE_LLM:
            if prompt is None or response is None:
                raise ValueError(
                    "The following fields cannot be None for GENERATIVE_LLM models: prompt, response"
                )
            for emb_name, emb_obj in {"prompt": prompt, "response": response}.items():
                # Must verify embedding type
                if not isinstance(emb_obj, Embedding):
                    raise TypeError("Both prompt and response objects must be of type Embedding")
                emb_obj.validate(emb_name)
        else:
            if prompt is not None or response is not None:
                raise ValueError(
                    "The fields 'prompt' and 'response' must be None for model types other "
                    "than GENERATIVE_LLM"
                )

        # Construct the prediction
        p = None
        if model_type == ModelTypes.GENERATIVE_LLM and prediction_label is None:
            prediction_label = 1
        if prediction_label is not None:
            if model_version is not None and not isinstance(model_version, str):
                raise TypeError(
                    f"model_version {model_version} is type {type(model_version)}, but must be a "
                    f"str"
                )
            _validate_label(
                prediction_or_actual="prediction",
                model_type=model_type,
                label=convert_element(prediction_label),
                embedding_features=embedding_features,
            )
            p = pb2.Prediction(
                prediction_label=_get_label(
                    prediction_or_actual="prediction",
                    value=prediction_label,
                    model_type=model_type,
                ),
                model_version=model_version,
            )
            if features is not None:
                converted_feats = convert_dictionary(features)
                feats = pb2.Prediction(features=converted_feats)
                p.MergeFrom(feats)

            if embedding_features or prompt or response:
                # NOTE: Deep copy is necessary to avoid side effects on the original input dictionary
                combined_embedding_features = {k: v for k, v in embedding_features.items()}
                # Map prompt/response as embedding features for generative models
                if model_type == ModelTypes.GENERATIVE_LLM:
                    combined_embedding_features.update({"prompt": prompt, "response": response})
                converted_embedding_feats = convert_dictionary(combined_embedding_features)
                embedding_feats = pb2.Prediction(features=converted_embedding_feats)
                p.MergeFrom(embedding_feats)

            if tags is not None:
                converted_tags = convert_dictionary(tags)
                tgs = pb2.Prediction(tags=converted_tags)
                p.MergeFrom(tgs)

            if prediction_timestamp is not None:
                p.timestamp.MergeFrom(get_timestamp(prediction_timestamp))

        # Validate and construct the optional actual
        a = None
        if actual_label is not None:
            _validate_label(
                prediction_or_actual="actual",
                model_type=model_type,
                label=convert_element(actual_label),
                embedding_features=embedding_features,
            )
            a = pb2.Actual(
                actual_label=_get_label(
                    prediction_or_actual="actual",
                    value=actual_label,
                    model_type=model_type,
                )
            )
            # Added to support delayed tags on actuals.
            if tags is not None:
                converted_tags = convert_dictionary(tags)
                tgs = pb2.Actual(tags=converted_tags)
                a.MergeFrom(tgs)

        # Validate and construct the optional feature importances
        fi = None
        if shap_values is not None and bool(shap_values):
            for k, v in shap_values.items():
                if not isinstance(convert_element(v), float):
                    raise TypeError(
                        f"feature {k} with value {v} is type {type(v)}, but expected one of: float"
                    )
                if isinstance(k, str) and k.endswith("_shap"):
                    raise ValueError(f"feature {k} must not be named with a `_shap` suffix")
            fi = pb2.FeatureImportances(feature_importances=shap_values)

        if p is None and a is None and fi is None:
            raise ValueError(
                "must provide at least one of prediction_label, actual_label, or shap_values"
            )

        env_params = None
        if environment == Environments.TRAINING:
            if p is None or a is None:
                raise ValueError("Training records must have both Prediction and Actual")
            env_params = pb2.Record.EnvironmentParams(
                training=pb2.Record.EnvironmentParams.Training()
            )
        elif environment == Environments.VALIDATION:
            if p is None or a is None:
                raise ValueError("Validation records must have both Prediction and Actual")
            env_params = pb2.Record.EnvironmentParams(
                validation=pb2.Record.EnvironmentParams.Validation(batch_id=batch_id)
            )
        elif environment == Environments.PRODUCTION:
            env_params = pb2.Record.EnvironmentParams(
                production=pb2.Record.EnvironmentParams.Production()
            )

        rec = pb2.Record(
            space_key=self._space_key,
            model_id=model_id,
            prediction_id=prediction_id,
            prediction=p,
            actual=a,
            feature_importances=fi,
            environment_params=env_params,
            is_generative_llm_record=BoolValue(value=model_type == ModelTypes.GENERATIVE_LLM),
        )
        return self._post(record=rec, uri=self._uri, indexes=None)

    def _post(self, record, uri, indexes):
        resp = self._session.post(
            uri,
            headers=self._headers,
            timeout=self._timeout,
            json=MessageToDict(message=record, preserving_proto_field_name=True),
        )
        if indexes is not None and len(indexes) == 2:
            resp.starting_index = indexes[0]
            resp.ending_index = indexes[1]
        return resp


def _validate_label(
    prediction_or_actual: str,
    model_type: ModelTypes,
    label: Union[
        str,
        bool,
        int,
        float,
        Tuple[Union[str, bool], float],
        ObjectDetectionLabel,
        RankingPredictionLabel,
        RankingActualLabel,
    ],
    embedding_features: Dict[str, Embedding],
):
    if model_type in NUMERIC_MODEL_TYPES:
        _validate_numeric_label(model_type, label)
    elif model_type in CATEGORICAL_MODEL_TYPES:
        _validate_categorical_label(model_type, label)
    elif model_type == ModelTypes.OBJECT_DETECTION:
        _validate_object_detection_label(prediction_or_actual, label, embedding_features)
    elif model_type == ModelTypes.RANKING:
        _validate_ranking_label(label)
    elif model_type == ModelTypes.GENERATIVE_LLM:
        _validate_generative_llm_label(label)
    else:
        raise TypeError(
            f"model_type {model_type} is type {type(model_type)}, but must be of "
            f"arize.utils.ModelTypes"
        )


def _validate_numeric_label(
    model_type: ModelTypes,
    label: Union[str, bool, int, float, Tuple[Union[str, bool], float]],
):
    if not isinstance(label, (float, int)):
        raise TypeError(
            f"label {label} has type {type(label)}, but must be either float or int for "
            f"{model_type}"
        )


def _validate_categorical_label(
    model_type: ModelTypes,
    label: Union[str, bool, int, float, Tuple[Union[str, bool], float]],
):
    is_valid = (
        isinstance(label, str)
        or isinstance(label, bool)
        or isinstance(label, int)
        or isinstance(label, float)
        or (
            isinstance(label, tuple)
            and isinstance(label[0], (str, bool))
            and isinstance(label[1], float)
        )
    )
    if not is_valid:
        raise TypeError(
            f"label {label} has type {type(label)}, but must be str, bool, int, float or Tuple[str, "
            f"float] for {model_type}"
        )


def _validate_object_detection_label(
    prediction_or_actual: str,
    label: ObjectDetectionLabel,
    embedding_features: Dict[str, Embedding],
):
    if not isinstance(label, ObjectDetectionLabel):
        raise TypeError(
            f"label {label} has type {type(label)}, but must be ObjectDetectionLabel for"
            f"{ModelTypes.OBJECT_DETECTION}"
        )
    if embedding_features is None:
        raise ValueError("Cannot use Object Detection Labels without an embedding feature")
    if len(embedding_features.keys()) != 1:
        raise ValueError("Object Detection Labels must be sent with exactly one embedding feature")
    label.validate(prediction_or_actual=prediction_or_actual)


def _validate_ranking_label(
    label: Union[RankingPredictionLabel, RankingActualLabel],
):
    if not isinstance(label, (RankingPredictionLabel, RankingActualLabel)):
        raise TypeError(
            f"label {label} has type {type(label)}, but must be RankingPredictionLabel"
            f"or RankingActualLabel for {ModelTypes.RANKING}"
        )
    label.validate()


def _validate_generative_llm_label(
    label: Union[str, bool, int, float],
):
    is_valid = (
        isinstance(label, str)
        or isinstance(label, bool)
        or isinstance(label, int)
        or isinstance(label, float)
    )
    if not is_valid:
        raise TypeError(
            f"label {label} has type {type(label)}, but must be str, bool, int, float "
            f"for {ModelTypes.GENERATIVE_LLM}"
        )


def _get_label(
    prediction_or_actual: str,
    value: Union[
        str,
        bool,
        int,
        float,
        Tuple[str, float],
        ObjectDetectionLabel,
        RankingPredictionLabel,
        RankingActualLabel,
    ],
    model_type: ModelTypes,
) -> Union[pb2.PredictionLabel, pb2.ActualLabel]:
    value = convert_element(value)
    if model_type in NUMERIC_MODEL_TYPES:
        return _get_numeric_label(prediction_or_actual, value)
    elif model_type in CATEGORICAL_MODEL_TYPES or model_type == ModelTypes.GENERATIVE_LLM:
        return _get_score_categorical_label(prediction_or_actual, value)
    elif model_type == ModelTypes.OBJECT_DETECTION:
        return _get_object_detection_label(prediction_or_actual, value)
    elif model_type == ModelTypes.RANKING:
        return _get_ranking_label(value)
    raise ValueError(
        f"model_type must be one of: {[mt.prediction_or_actual for mt in ModelTypes]} "
        f"Got "
        f"{model_type} instead."
    )


def _get_numeric_label(
    prediction_or_actual: str,
    value: Union[int, float],
) -> Union[pb2.PredictionLabel, pb2.ActualLabel]:
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"Received {prediction_or_actual}_label = {value}, of type {type(value)}. "
            + f"{[mt.prediction_or_actual for mt in NUMERIC_MODEL_TYPES]} models accept labels of "
            f"type int or float"
        )
    if prediction_or_actual == "prediction":
        return pb2.PredictionLabel(numeric=value)
    elif prediction_or_actual == "actual":
        return pb2.ActualLabel(numeric=value)


def _get_score_categorical_label(
    prediction_or_actual: str,
    value: Union[bool, str, Tuple[str, float]],
) -> Union[pb2.PredictionLabel, pb2.ActualLabel]:
    sc = pb2.ScoreCategorical()
    if isinstance(value, bool):
        sc.category.category = str(value)
    elif isinstance(value, str):
        sc.category.category = value
    elif isinstance(value, int) or isinstance(value, float):
        sc.score_value.value = value
    elif isinstance(value, tuple):
        # Expect Tuple[str,float]
        if value[1] is None:
            raise TypeError(
                f"Received {prediction_or_actual}_label = {value}, of type "
                f"{type(value)}[{type(value[0])}, None]. "
                f"{[mt.prediction_or_actual for mt in CATEGORICAL_MODEL_TYPES]} models accept "
                "values of type str, bool, or Tuple[str, float]"
            )
        if not isinstance(value[0], (bool, str)) or not isinstance(value[1], float):
            raise TypeError(
                f"Received {prediction_or_actual}_label = {value}, of type "
                f"{type(value)}[{type(value[0])}, {type(value[1])}]. "
                f"{[mt.prediction_or_actual for mt in CATEGORICAL_MODEL_TYPES]} models accept "
                "values of type str, bool, or Tuple[str or bool, float]"
            )
        if isinstance(value[0], bool):
            sc.score_category.category = str(value[0])
        else:
            sc.score_category.category = value[0]
        sc.score_category.score = value[1]
    else:
        raise TypeError(
            f"Received {prediction_or_actual}_label = {value}, of type {type(value)}. "
            + f"{[mt.prediction_or_actual for mt in CATEGORICAL_MODEL_TYPES]} models accept values "
            f"of type str, bool, int, float or Tuple[str, float]"
        )
    if prediction_or_actual == "prediction":
        return pb2.PredictionLabel(score_categorical=sc)
    elif prediction_or_actual == "actual":
        return pb2.ActualLabel(score_categorical=sc)


def _get_object_detection_label(
    prediction_or_actual: str,
    value: ObjectDetectionLabel,
) -> Union[pb2.PredictionLabel, pb2.ActualLabel]:
    if not isinstance(value, ObjectDetectionLabel):
        raise TypeError(
            f"label {value} has type {type(value)}, but must be ObjectDetectionLabel for ModelTypes"
            f".OBJECT_DETECTION"
        )
    od = pb2.ObjectDetection()
    bounding_boxes = []
    for i in range(len(value.bounding_boxes_coordinates)):
        coordinates = value.bounding_boxes_coordinates[i]
        category = value.categories[i]
        if value.scores is None:
            bounding_boxes.append(
                pb2.ObjectDetection.BoundingBox(coordinates=coordinates, category=category)
            )
        else:
            score = value.scores[i]
            bounding_boxes.append(
                pb2.ObjectDetection.BoundingBox(
                    coordinates=coordinates,
                    category=category,
                    score=DoubleValue(value=score),
                )
            )

    od.bounding_boxes.extend(bounding_boxes)
    if prediction_or_actual == "prediction":
        return pb2.PredictionLabel(object_detection=od)
    elif prediction_or_actual == "actual":
        return pb2.ActualLabel(object_detection=od)


def _get_ranking_label(
    value: Union[RankingPredictionLabel, RankingActualLabel]
) -> Union[pb2.PredictionLabel, pb2.ActualLabel]:
    if not isinstance(value, (RankingPredictionLabel, RankingActualLabel)):
        raise TypeError(
            f"label {value} has type {type(value)}, but must be RankingPredictionLabel"
            f" or RankingActualLabel for ModelTypes.RANKING"
        )
    if isinstance(value, RankingPredictionLabel):
        rp = pb2.RankingPrediction()
        # If validation has passed, rank and group_id are guaranteed to be not None
        rp.rank = value.rank
        rp.prediction_group_id = value.group_id
        # score and label are optional
        if value.score is not None:
            rp.prediction_score.value = value.score
        if value.label is not None:
            rp.label = value.label
        return pb2.PredictionLabel(ranking=rp)
    elif isinstance(value, RankingActualLabel):
        ra = pb2.RankingActual()
        # relevance_labels and relevance_score are optional
        if value.relevance_labels is not None:
            ra.category.values.extend(value.relevance_labels)
        if value.relevance_score is not None:
            ra.relevance_score.value = value.relevance_score
        return pb2.ActualLabel(ranking=ra)


def _validate_mapping_key(feat_name):
    if not isinstance(feat_name, str):
        raise ValueError(
            f"feature {feat_name} must be named with string, type used: {type(feat_name)}"
        )
    if feat_name.endswith("_shap"):
        raise ValueError(f"feature {feat_name} must not be named with a `_shap` suffix")
    return


def _convert_prediction_id(prediction_id: Union[str, int, float]) -> str:
    if not isinstance(prediction_id, str):
        try:
            prediction_id = str(
                prediction_id
            ).strip()  # strip ensures we don't receive whitespaces as part of the prediction id
        except Exception:
            raise ValueError(f"prediction_id value {prediction_id} must be convertible to a string")

    if len(prediction_id) not in range(MIN_PREDICTION_ID_LEN, MAX_PREDICTION_ID_LEN + 1):
        raise ValueError(
            f"The string length of prediction_id {prediction_id} must be between {MIN_PREDICTION_ID_LEN} "
            f"and {MAX_PREDICTION_ID_LEN}"
        )
    return prediction_id


def _validate_and_convert_prediction_id(
    prediction_id: Optional[Union[str, int, float]],
    environment: Environments,
    prediction_label: Optional[PredictionLabelTypes] = None,
    actual_label: Optional[ActualLabelTypes] = None,
    shap_values: Optional[Dict[str, float]] = None,
) -> str:
    # If the user does not provide prediction id
    if prediction_id is None:
        # delayed records have actual information but not prediction information
        is_delayed_record = prediction_label is None and (
            actual_label is not None or shap_values is not None
        )
        # Pre-production environment does not need prediction id
        # Production environment needs prediction id for delayed record, since joins are needed
        if is_delayed_record and environment == Environments.PRODUCTION:
            raise ValueError(
                "prediction_id value cannot be None for delayed records, i.e., records ",
                "without prediction_label and with either actual_label or shap_values",
            )
        # Prediction ids are optional for:
        # pre-production records and
        # production records that are not delayed records
        return ""

    # If prediction id is given by user, convert it to string and validate length
    return _convert_prediction_id(prediction_id)
