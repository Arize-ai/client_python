# type: ignore[pb2]
import concurrent.futures as cf
import copy
import os
import time
from typing import Dict, List, Optional, Tuple, Union

from arize.pandas.validation.errors import InvalidAdditionalHeaders, InvalidNumberOfEmbeddings
from arize.utils.constants import (
    API_KEY_ENVVAR_NAME,
    LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_LLM_MODEL_NAME_LENGTH,
    MAX_LLM_MODEL_NAME_LENGTH_TRUNCATION,
    MAX_NUMBER_OF_EMBEDDINGS,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
    MAX_PREDICTION_ID_LEN,
    MAX_PROMPT_TEMPLATE_LENGTH,
    MAX_PROMPT_TEMPLATE_LENGTH_TRUNCATION,
    MAX_PROMPT_TEMPLATE_VERSION_LENGTH,
    MAX_PROMPT_TEMPLATE_VERSION_LENGTH_TRUNCATION,
    MAX_TAG_LENGTH,
    MAX_TAG_LENGTH_TRUNCATION,
    MIN_PREDICTION_ID_LEN,
    RESERVED_TAG_COLS,
    SPACE_ID_ENVVAR_NAME,
    SPACE_KEY_ENVVAR_NAME,
)
from google.protobuf.json_format import MessageToDict
from google.protobuf.wrappers_pb2 import BoolValue, DoubleValue
from requests_futures.sessions import FuturesSession

from . import public_pb2 as pb2
from .__init__ import __version__
from .bounded_executor import BoundedExecutor
from .single_log.casting import cast_dictionary
from .utils.errors import (
    AuthError,
    InvalidCertificateFile,
    InvalidStringLength,
    InvalidTypeAuthKey,
    InvalidValueType,
)
from .utils.logging import get_truncation_warning_message, logger
from .utils.types import (
    CATEGORICAL_MODEL_TYPES,
    NUMERIC_MODEL_TYPES,
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
    _PromptOrResponseText,
    is_list_of,
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
    MultiClassPredictionLabel,
]
ActualLabelTypes = Union[
    str,
    bool,
    int,
    float,
    Tuple[str, float],
    ObjectDetectionLabel,
    RankingActualLabel,
    MultiClassActualLabel,
]


class Client:
    """
    Arize API Client to log model predictions and actuals to the Arize AI platform
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        space_id: Optional[str] = None,
        space_key: Optional[str] = None,
        uri: Optional[str] = "https://api.arize.com/v1",
        max_workers: Optional[int] = 8,
        max_queue_bound: Optional[int] = 5000,
        timeout: Optional[int] = 200,
        additional_headers: Optional[Dict[str, str]] = None,
        request_verify: Union[bool, str] = True,
    ) -> None:
        """
        Initializes the Arize Client

        Arguments:
        ----------
            api_key (str): Arize provided API key associated with your account. Located on the
                data upload page.
            space_id (str): Arize provided identifier to connect records to spaces. Located on
                the space settings.
            space_key (str): [Deprecated] Arize provided identifier to connect records to spaces.
                Located on the space settings.
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
        space_id = space_id or os.getenv(SPACE_ID_ENVVAR_NAME)
        space_key = space_key or os.getenv(SPACE_KEY_ENVVAR_NAME)

        if space_key is not None:
            logger.warning(
                "The space_key parameter is deprecated and will be removed in a future release. "
                "Please use the space_id parameter instead."
            )
        # api_key and one of space_id or space_key must be provided
        both_space_id_and_key_missing = space_id is None and space_key is None
        if api_key is None or both_space_id_and_key_missing:
            raise AuthError(api_key=api_key, space_key=space_key, space_id=space_id)
        # Check if the provided keys are of the correct type
        if any(not isinstance(key, str) for key in [api_key, space_key, space_id] if key):
            raise InvalidTypeAuthKey(api_key=api_key, space_key=space_key, space_id=space_id)
        if isinstance(request_verify, str) and not os.path.isfile(request_verify):
            raise InvalidCertificateFile(request_verify)
        self._request_verify = request_verify
        self._api_key = api_key
        self._space_key = space_key
        self._space_id = space_id
        self._uri = f"{uri}/log"
        self._timeout = timeout
        self._session = FuturesSession(executor=BoundedExecutor(max_queue_bound, max_workers))
        # Grpc-Metadata prefix is required to pass non-standard md through via grpc-gateway
        self._headers = {
            "authorization": api_key,
            "Grpc-Metadata-space": space_key,
            "Grpc-Metadata-space_id": space_id,
            "Grpc-Metadata-sdk-language": "python",
            "Grpc-Metadata-language-version": get_python_version(),
            "Grpc-Metadata-sdk-version": __version__,
        }
        if additional_headers is not None:
            conflicting_keys = self._headers.keys() & additional_headers.keys()
            if conflicting_keys:
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
        features: Optional[Dict[str, Union[str, bool, float, int, List[str], TypedValue]]] = None,
        embedding_features: Optional[Dict[str, Embedding]] = None,
        shap_values: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, Union[str, bool, float, int, TypedValue]]] = None,
        batch_id: Optional[str] = None,
        prompt: Optional[Union[str, Embedding]] = None,
        response: Optional[Union[str, Embedding]] = None,
        prompt_template: Optional[str] = None,
        prompt_template_version: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_params: Optional[Dict[str, Union[str, bool, float, int]]] = None,
        llm_run_metadata: Optional[LLMRunMetadata] = None,
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
            prediction_label (bool, int, float, str, Tuple(str, float), ObjectDetectionLabel,
                RankingPredictionLabel or MultiClassPredictionLabel; optional):
                The predicted value for a given model input. Defaults to None.
            actual_label (bool, int, float, str, Tuple[str, float], ObjectDetectionLabel,
                RankingActualLabel or MultiClassActualLabel; optional):
                The ground truth value for a given model input. This will be matched to the
                prediction with the same prediction_id as the one in this call. Defaults to None.
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
            prompt (str or Embedding, optional): input text on which the GENERATIVE_LLM model acts. It
                accepts a string or Embedding object (if sending embedding vectors is desired). Required
                for GENERATIVE_LLM models. Defaults to None.
            response (str or Embedding, optional): output text from GENERATIVE_LLM model. It accepts
                a string or Embedding object (if sending embedding vectors is desired). Required for
                GENERATIVE_LLM models. Defaults to None.
            prompt_template (str, optional): template used to construct the prompt passed to a large language
                model. It can include variable using the double braces notation. Example: 'Given the context
                {{context}}, answer the following question {{user_question}}.
            prompt_template_version (str, optional): version of the template used.
            llm_model_name (str, optional): name of the llm used. Example: 'gpt-4'.
            llm_params (str, optional): hyperparameters passed to the large language model. Example:
                {
                   "temperature": 0.9,
                   "presence_penalty": 0.34,
                   "stop": [".", "?", "!"],
                }
            llm_run_metadata (LLMRunMetadata, optional): run metadata for llm calls. Example:
                LLMRunMetadata(
                   total_token_count=400,
                   prompt_token_count=300,
                   response_token_count=100,
                   response_latency_ms=2000,
                )

        Returns:
        --------
            `concurrent.futures.Future` object
        """
        # Validate model_id
        if not isinstance(model_id, str):
            raise InvalidValueType("model_id", model_id, "str")
        # Validate model_type
        if not isinstance(model_type, ModelTypes):
            raise InvalidValueType("model_type", model_type, "arize.utils.ModelTypes")
        # Validate environment
        if not isinstance(environment, Environments):
            raise InvalidValueType("environment", environment, "arize.utils.Environments")
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

        # Cast feature & tag values
        features = cast_dictionary(features)
        tags = cast_dictionary(tags)

        # Validate feature types
        if features:
            if not isinstance(features, dict):
                raise InvalidValueType("features", features, "dict")
            for feat_name, feat_value in features.items():
                _validate_mapping_key(feat_name, "features")
                if is_list_of(feat_value, str):
                    continue
                else:
                    val = convert_element(feat_value)
                    if val is not None and not isinstance(val, (str, bool, float, int)):
                        raise InvalidValueType(
                            f"feature '{feat_name}'", feat_value, "one of: bool, int, float, str"
                        )

        # Validate embedding_features type
        if embedding_features:
            if not isinstance(embedding_features, dict):
                raise InvalidValueType("embedding_features", embedding_features, "dict")
            if len(embedding_features) > MAX_NUMBER_OF_EMBEDDINGS:
                raise InvalidNumberOfEmbeddings(len(embedding_features))
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
                _validate_mapping_key(emb_name, "embedding features")
                # Must verify embedding type
                if not isinstance(emb_obj, Embedding):
                    raise InvalidValueType(f"embedding feature '{emb_name}'", emb_obj, "Embedding")
                emb_obj.validate(emb_name)

        # Validate tag types
        if tags:
            if not isinstance(tags, dict):
                raise InvalidValueType("tags", tags, "dict")
            wrong_tags = [tag_name for tag_name in tags.keys() if tag_name in RESERVED_TAG_COLS]
            if wrong_tags:
                raise KeyError(
                    f"The following tag names are not allowed as they are reserved: {wrong_tags}"
                )
            for tag_name, tag_value in tags.items():
                _validate_mapping_key(tag_name, "tags")
                val = convert_element(tag_value)
                if val is not None and not isinstance(val, (str, bool, float, int)):
                    raise InvalidValueType(
                        f"tag '{tag_name}'", tag_value, "one of: bool, int, float, str"
                    )
                if isinstance(tag_name, str) and tag_name.endswith("_shap"):
                    raise ValueError(f"tag {tag_name} must not be named with a `_shap` suffix")
                if len(str(val)) > MAX_TAG_LENGTH:
                    raise ValueError(
                        f"The number of characters for each tag must be less than or equal to "
                        f"{MAX_TAG_LENGTH}. The tag {tag_name} with value {tag_value} has "
                        f"{len(str(val))} characters."
                    )
                elif len(str(val)) > MAX_TAG_LENGTH_TRUNCATION:
                    logger.warning(
                        get_truncation_warning_message("tags", MAX_TAG_LENGTH_TRUNCATION)
                    )

        # Check the timestamp present on the event
        if prediction_timestamp is not None:
            if not isinstance(prediction_timestamp, int):
                raise InvalidValueType("prediction_timestamp", prediction_timestamp, "int")
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
            if prompt is not None:
                if not isinstance(prompt, (str, Embedding)):
                    raise TypeError(
                        f"prompt must be of type str or Embedding, but found {type(val)}"
                    )
                if isinstance(prompt, str):
                    prompt = _PromptOrResponseText(data=prompt)
                # Validate content of prompt, type is either Embedding or _PromptOrResponseText
                prompt.validate("prompt")
            if response is not None:
                if not isinstance(response, (str, Embedding)):
                    raise TypeError(
                        f"response must be of type str or Embedding, but found {type(val)}"
                    )
                if isinstance(response, str):
                    response = _PromptOrResponseText(data=response)
                # Validate content of response, type is either Embedding or _PromptOrResponseText
                response.validate("response")

            # Validate prompt templates workflow information
            _validate_prompt_templates_and_llm_config(
                prompt_template, prompt_template_version, llm_model_name, llm_params
            )
            # Validate llm run metadata
            if llm_run_metadata is not None:
                llm_run_metadata.validate()
        else:
            if prompt is not None or response is not None:
                raise ValueError(
                    "The fields 'prompt' and 'response' must be None for model types other "
                    "than GENERATIVE_LLM"
                )

        # Construct the prediction
        p = None
        # Only set a default prediction label for generative LLM models if there is no explicit prediction
        # label AND no actual label to ensure that generative LLM model prediction records will have a
        # prediction label that can be used for metrics in the platform (as users will generally pass in
        # actuals/user feedback only). We do not want to add the default prediction label in if actual labels
        # are present so that latent actuals will still work.
        if (
            model_type == ModelTypes.GENERATIVE_LLM
            and prediction_label is None
            and actual_label is None
        ):
            prediction_label = "1"

        if prediction_label is not None:
            if model_version is not None and not isinstance(model_version, str):
                raise InvalidValueType("model_version", model_version, "str")
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
                combined_embedding_features = (
                    {k: v for k, v in embedding_features.items()} if embedding_features else {}
                )
                # Map prompt as embedding features for generative models
                if prompt is not None:
                    combined_embedding_features.update({"prompt": prompt})
                # Map response as embedding features for generative models
                if response is not None:
                    combined_embedding_features.update({"response": response})
                converted_embedding_feats = convert_dictionary(combined_embedding_features)
                embedding_feats = pb2.Prediction(features=converted_embedding_feats)
                p.MergeFrom(embedding_feats)

            if tags or llm_run_metadata:
                joined_tags = copy.deepcopy(tags)
                if llm_run_metadata:
                    if llm_run_metadata.total_token_count is not None:
                        joined_tags[
                            LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME
                        ] = llm_run_metadata.total_token_count
                    if llm_run_metadata.prompt_token_count is not None:
                        joined_tags[
                            LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME
                        ] = llm_run_metadata.prompt_token_count
                    if llm_run_metadata.response_token_count is not None:
                        joined_tags[
                            LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME
                        ] = llm_run_metadata.response_token_count
                    if llm_run_metadata.response_latency_ms is not None:
                        joined_tags[
                            LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME
                        ] = llm_run_metadata.response_latency_ms
                converted_tags = convert_dictionary(joined_tags)
                tgs = pb2.Prediction(tags=converted_tags)
                p.MergeFrom(tgs)

            if prompt_template or prompt_template_version or llm_model_name or llm_params:
                llm_fields = pb2.LLMFields(
                    prompt_template=prompt_template or "",
                    prompt_template_name=prompt_template_version or "",
                    llm_model_name=llm_model_name or "",
                    llm_params=convert_dictionary(llm_params),
                )
                p.MergeFrom(pb2.Prediction(llm_fields=llm_fields))

            if prediction_timestamp is not None:
                p.timestamp.MergeFrom(get_timestamp(prediction_timestamp))

        # Validate and construct the optional actual
        is_latent_tags = prediction_label is None and tags is not None
        a = None
        if actual_label or is_latent_tags:
            a = pb2.Actual()
            if actual_label is not None:
                _validate_label(
                    prediction_or_actual="actual",
                    model_type=model_type,
                    label=convert_element(actual_label),
                    embedding_features=embedding_features,
                )
                a.MergeFrom(
                    pb2.Actual(
                        actual_label=_get_label(
                            prediction_or_actual="actual",
                            value=actual_label,
                            model_type=model_type,
                        )
                    )
                )
            # Added to support delayed tags on actuals.
            if tags is not None:
                converted_tags = convert_dictionary(tags)
                a.MergeFrom(pb2.Actual(tags=converted_tags))

        # Validate and construct the optional feature importances
        fi = None
        if shap_values is not None and bool(shap_values):
            for k, v in shap_values.items():
                if not isinstance(convert_element(v), float):
                    raise InvalidValueType(f"feature '{k}'", v, "float")
                if isinstance(k, str) and k.endswith("_shap"):
                    raise ValueError(f"feature {k} must not be named with a `_shap` suffix")
            fi = pb2.FeatureImportances(feature_importances=shap_values)

        if p is None and a is None and fi is None:
            raise ValueError(
                "must provide at least one of prediction_label, actual_label, tags, or shap_values"
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
            verify=self._request_verify,
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
        MultiClassPredictionLabel,
        MultiClassActualLabel,
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
    elif model_type == ModelTypes.MULTI_CLASS:
        _validate_multi_class_label(label)
    else:
        raise InvalidValueType("model_type", model_type, "arize.utils.ModelTypes")


def _validate_numeric_label(
    model_type: ModelTypes,
    label: Union[str, bool, int, float, Tuple[Union[str, bool], float]],
):
    if not isinstance(label, (float, int)):
        raise InvalidValueType(
            f"label {label}", label, f"either float or int for model_type {model_type}"
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
        raise InvalidValueType(
            f"label {label}",
            label,
            f"one of: bool, int, float, str or Tuple[str, float] for model type {model_type}",
        )


def _validate_object_detection_label(
    prediction_or_actual: str,
    label: ObjectDetectionLabel,
    embedding_features: Dict[str, Embedding],
):
    if not isinstance(label, ObjectDetectionLabel):
        raise InvalidValueType(
            f"label {label}",
            label,
            f"ObjectDetectionLabel for model type {ModelTypes.OBJECT_DETECTION}",
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
        raise InvalidValueType(
            f"label {label}",
            label,
            f"RankingPredictionLabel or RankingActualLabel for model type {ModelTypes.RANKING}",
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
        raise InvalidValueType(
            f"label {label}",
            label,
            f"one of: bool, int, float, str for model type {ModelTypes.GENERATIVE_LLM}",
        )


def _validate_multi_class_label(
    label: Union[MultiClassPredictionLabel, MultiClassActualLabel],
):
    if not isinstance(label, (MultiClassPredictionLabel, MultiClassActualLabel)):
        raise InvalidValueType(
            f"label {label}",
            label,
            f"MultiClassPredictionLabel or MultiClassActualLabel for model type {ModelTypes.MULTI_CLASS}",
        )
    label.validate()


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
        MultiClassPredictionLabel,
        MultiClassActualLabel,
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
    elif model_type == ModelTypes.MULTI_CLASS:
        return _get_multi_class_label(value)
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
        raise InvalidValueType(
            "object detection label",
            value,
            f"ObjectDetectionLabel for model type {ModelTypes.OBJECT_DETECTION}",
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
        raise InvalidValueType(
            "ranking label",
            value,
            f"RankingPredictionLabel or RankingActualLabel for model type {ModelTypes.RANKING}",
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


def _get_multi_class_label(
    value: Union[MultiClassPredictionLabel, MultiClassActualLabel]
) -> Union[pb2.PredictionLabel, pb2.ActualLabel]:
    if not isinstance(value, (MultiClassPredictionLabel, MultiClassActualLabel)):
        raise InvalidValueType(
            "multi class label",
            value,
            f"MultiClassPredictionLabel or MultiClassActualLabel for model type {ModelTypes.MULTI_CLASS}",
        )
    if isinstance(value, MultiClassPredictionLabel):
        mc_pred = pb2.MultiClassPrediction()
        # threshold score map is not None in multi-label case
        if value.threshold_scores is not None:
            prediction_threshold_scores = {}
            # Validations checked prediction score map is not None
            for class_name, p_score in value.prediction_scores.items():
                # Validations checked threshold map contains all classes so safe to index w class_name
                multi_label_scores = pb2.MultiClassPrediction.MultiLabel.MultiLabelScores(
                    prediction_score=DoubleValue(value=p_score),
                    threshold_score=DoubleValue(value=value.threshold_scores[class_name]),
                )
                prediction_threshold_scores[class_name] = multi_label_scores
            multi_label = pb2.MultiClassPrediction.MultiLabel(
                prediction_threshold_scores=prediction_threshold_scores
            )
            mc_pred = pb2.MultiClassPrediction(multi_label=multi_label)
        else:
            prediction_scores_double_values = {}
            # Validations checked prediction score map is not None
            for class_name, p_score in value.prediction_scores.items():
                prediction_scores_double_values[class_name] = DoubleValue(value=p_score)
            single_label = pb2.MultiClassPrediction.SingleLabel(
                prediction_scores=prediction_scores_double_values,
            )
            mc_pred = pb2.MultiClassPrediction(single_label=single_label)
        p_label = pb2.PredictionLabel(multi_class=mc_pred)
        return p_label
    elif isinstance(value, MultiClassActualLabel):
        # Validations checked actual score map is not None
        actual_labels = []  # list of class names with actual score of 1
        for class_name, score in value.actual_scores.items():
            if float(score) == 1.0:
                actual_labels.append(class_name)
        mc_act = pb2.MultiClassActual(
            actual_labels=actual_labels,
        )
        return pb2.ActualLabel(multi_class=mc_act)


def _validate_mapping_key(key_name: str, name: str):
    if not isinstance(key_name, str):
        raise ValueError(
            f"{name} dictionary key {key_name} must be named with string, type used: {type(key_name)}"
        )
    if key_name.endswith("_shap"):
        raise ValueError(
            f"{name} dictionary key {key_name} must not be named with a `_shap` suffix"
        )
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


def _validate_prompt_templates_and_llm_config(
    prompt_template: Optional[str],
    prompt_template_version: Optional[str],
    llm_model_name: Optional[str],
    llm_params: Optional[Dict[str, Union[str, bool, float, int]]],
) -> None:
    if prompt_template is not None:
        if not isinstance(prompt_template, str):
            raise InvalidValueType(
                "prompt_template",
                prompt_template,
                "str",
            )
        if len(prompt_template) > MAX_PROMPT_TEMPLATE_LENGTH:
            raise InvalidStringLength("prompt_template", 0, MAX_PROMPT_TEMPLATE_LENGTH)
        elif len(prompt_template) > MAX_PROMPT_TEMPLATE_LENGTH_TRUNCATION:
            logger.warning(
                get_truncation_warning_message(
                    "prompt templates", MAX_PROMPT_TEMPLATE_LENGTH_TRUNCATION
                )
            )

    if prompt_template_version is not None:
        if not isinstance(prompt_template_version, str):
            raise InvalidValueType(
                "prompt_template_version",
                prompt_template_version,
                "str",
            )
        if len(prompt_template_version) > MAX_PROMPT_TEMPLATE_VERSION_LENGTH:
            raise InvalidStringLength(
                "prompt_template_version", 0, MAX_PROMPT_TEMPLATE_VERSION_LENGTH
            )
        elif len(prompt_template_version) > MAX_PROMPT_TEMPLATE_VERSION_LENGTH_TRUNCATION:
            logger.warning(
                get_truncation_warning_message(
                    "prompt template versions", MAX_PROMPT_TEMPLATE_VERSION_LENGTH_TRUNCATION
                )
            )

    if llm_model_name is not None:
        if not isinstance(llm_model_name, str):
            raise InvalidValueType(
                "llm_model_name",
                llm_model_name,
                "str",
            )
        if len(llm_model_name) > MAX_LLM_MODEL_NAME_LENGTH:
            raise InvalidStringLength("llm_model_name", 0, MAX_LLM_MODEL_NAME_LENGTH)
        elif len(llm_model_name) > MAX_LLM_MODEL_NAME_LENGTH_TRUNCATION:
            logger.warning(
                get_truncation_warning_message(
                    "LLM model names", MAX_LLM_MODEL_NAME_LENGTH_TRUNCATION
                )
            )

    if llm_params is not None:
        if not isinstance(llm_params, dict):
            raise InvalidValueType(
                "llm_params",
                llm_params,
                "dict",
            )
        for param_name, param_value in llm_params.items():
            _validate_mapping_key(param_name, "llm_params")
            val = convert_element(param_value)
            if val is None:
                continue
            is_correct_type = isinstance(val, (bool, int, float, str)) or is_list_of(val, str)
            if not is_correct_type:
                raise InvalidValueType(
                    f"llm param '{param_name}'",
                    param_value,
                    "one of: bool, int, float, str, list[str]",
                )
