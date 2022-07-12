import time

import pandas as pd
import numpy as np
import concurrent.futures as cf
from typing import Union, Optional, Dict, Tuple
from google.protobuf.json_format import MessageToDict
from requests_futures.sessions import FuturesSession

from arize import public_pb2 as public__pb2
from arize.bounded_executor import BoundedExecutor
from arize.utils.types import ModelTypes, Embedding
from arize.utils.utils import (
    validate_prediction_timestamps,
    convert_element,
    get_value_object,
    get_timestamp,
    is_timestamp_in_range,
    infer_model_type,
)

from arize.__init__ import __version__


class Client:
    """
    Arize API Client to report model predictions and actuals to Arize AI platform
    """

    def __init__(
        self,
        api_key: str,
        space_key: str,
        uri="https://api.arize.com/v1",
        max_workers=8,
        max_queue_bound=5000,
        retry_attempts=3,
        timeout=200,
    ):
        """
        :params api_key: (str) api key associated with your account with Arize AI
        :params space_key: (str) space key in Arize AI
        :params max_workers: (int) number of max concurrent requests to Arize. Default: 8
        :max_queue_bound: (int) number of maximum concurrent future objects being generated for publishing to Arize. Default: 5000
        """

        if not isinstance(space_key, str):
            raise TypeError(
                f"space_key {space_key} is type {type(space_key)}, but must be a str"
            )
        self._retry_attempts = retry_attempts
        self._uri = uri + "/log"
        self._bulk_url = uri + "/bulk"
        self._stream_uri = uri + "/preprod"
        self._files_uri = uri + "/files"
        self._api_key = api_key
        self._space_key = space_key
        self._timeout = timeout
        self._session = FuturesSession(
            executor=BoundedExecutor(max_queue_bound, max_workers)
        )
        # Grpc-Metadata prefix is required to pass non-standard md through via grpc-gateway
        self._header = {
            "authorization": api_key,
            "Grpc-Metadata-space": space_key,
            "Grpc-Metadata-sdk-version": __version__,
            "Grpc-Metadata-sdk": "py",
        }

    def log(
        self,
        model_id: str,
        prediction_id: Union[str, int, float],
        model_version: str = None,
        prediction_label: Union[str, bool, int, float, Tuple[str, float]] = None,
        actual_label: Union[str, bool, int, float, Tuple[str, float]] = None,
        shap_values: Dict[str, float] = None,
        features: Optional[
            Dict[Union[str, int, float], Union[str, bool, float, int]]
        ] = None,
        tags: Optional[
            Dict[Union[str, int, float], Union[str, bool, float, int]]
        ] = None,
        model_type: Optional[ModelTypes] = None,
        prediction_timestamp: Optional[int] = None,
        embedding_features: Optional[Dict[Union[str, int, float], Embedding]] = None,
    ) -> cf.Future:
        """Logs a record to Arize via a POST request. Returns :class:`Future` object.
        :param model_id: (str) Unique identifier for a given model
        :param prediction_id: (str, int, float) Unique string identifier for a specific prediction. This value is used to match a prediction to an actual label or feature imporances in the Arize platform.
        :param model_version: (str) Field used to group together a subset of predictions and actuals for a given model_id.
        :param prediction_label: (one of str, bool, int, float, Tuple[str, float]) The predicted value for a given model input.
        :param actual_label: (one of str, bool, int, float) The actual true value for a given model input. This actual will be matched to the prediction with the same prediction_id as the one in this call.
        :param shap_values: (str, float) Dictionary containing human readable and debuggable model features keys, along with SHAP feature importance values. Keys must be str, while values must be float.
        :param features: ((str, int, float), <value>) Optional dictionary containing human readable and debuggable model features. Keys must be one of str, int, or float. Values must be one of str, bool, float, long.
        :param tags: ((str, int, float), <value>) Optional dictionary containing human readable and debuggable model tags. Keys must be str, values one of str, bool, float, long.
        :param model_type: (ModelTypes) Declares what model type this prediction is for. Binary, Numeric, Categorical, Score_Categorical.
        :param prediction_timestamp: (int) Optional field with unix epoch time in seconds to overwrite timestamp for prediction. If None, prediction uses current timestamp.
        :param embedding_features ((str, int, float), Embedding): Optional dictionary containing model embedding features. Keys must be one of str, int, or float. Values must be of Embedding type.
        :rtype : concurrent.futures.Future
        """

        # Validate model_id
        if not isinstance(model_id, str):
            raise TypeError(
                f"model_id {model_id} is type {type(model_id)}, but must be a str"
            )

        # Validate feature types
        if features:
            for k, v in features.items():
                val = convert_element(v)
                if val is not None and not isinstance(val, (str, bool, float, int)):
                    raise TypeError(
                        f"feature {k} with value {v} is type {type(v)}, but expected one of: str, bool, float, int"
                    )
                if isinstance(k, str) and k.endswith("_shap"):
                    raise ValueError(
                        f"feature {k} must not be named with a `_shap` suffix"
                    )

        # Validate embedding_features type
        if embedding_features:
            for emb_name, emb_obj in embedding_features.items():
                # Must verify embedding type
                if type(emb_obj) != Embedding:
                    raise TypeError(
                        f'Embedding feature "{emb_name}" must be of embedding type'
                    )
                Embedding.validate_embedding_object(emb_name, emb_obj)

        # Validate tag types
        if tags:
            for k, v in tags.items():
                val = convert_element(v)
                if val is not None and not isinstance(val, (str, bool, float, int)):
                    raise TypeError(
                        f"tag {k} with value {v} is type {type(v)}, but expected one of: str, bool, float, int"
                    )
                if isinstance(k, str) and k.endswith("_shap"):
                    raise ValueError(f"tag {k} must not be named with a `_shap` suffix")

        # Check the timestamp present on the event
        if prediction_timestamp is not None and not isinstance(
            prediction_timestamp, int
        ):
            raise TypeError(
                f"prediction_timestamp {prediction_timestamp} is type {type(prediction_timestamp)} but expected int"
            )

        now = int(time.time())
        if prediction_timestamp is not None and not is_timestamp_in_range(
            now, prediction_timestamp
        ):
            raise ValueError(
                f"prediction_timestamp: {prediction_timestamp} is out of range. Value must be within 1 year of the current time."
            )

        # Construct the prediction
        p = None
        if prediction_label is not None:
            if not isinstance(model_version, str):
                raise TypeError(
                    f"model_version {model_version} is type {type(model_version)}, but must be a str"
                )
            model_type = (
                infer_model_type(prediction_label) if model_type is None else model_type
            )
            _label_validation(model_type, label=convert_element(prediction_label))
            p = public__pb2.Prediction(
                label=_get_label(
                    value=prediction_label,
                    name="prediction",
                    model_type=model_type,
                ),
                model_version=model_version,
            )
            if features is not None:
                converted_feats = {}
                for (k, v) in features.items():
                    val = get_value_object(value=v, name=k)
                    if val is not None:
                        converted_feats[str(k)] = val
                feats = public__pb2.Prediction(features=converted_feats)
                p.MergeFrom(feats)

            if embedding_features is not None:
                converted_embedding_feats = {}
                for (k, v) in embedding_features.items():
                    val = get_value_object(value=v, name=k)
                    if val is not None:
                        converted_embedding_feats[str(k)] = val
                embedding_feats = public__pb2.Prediction(
                    features=converted_embedding_feats
                )
                p.MergeFrom(embedding_feats)

            if tags is not None:
                converted_tags = {}
                for (k, v) in tags.items():
                    val = get_value_object(value=v, name=k)
                    if val is not None:
                        converted_tags[str(k)] = val
                tgs = public__pb2.Prediction(tags=converted_tags)
                p.MergeFrom(tgs)

            if prediction_timestamp is not None:
                p.timestamp.MergeFrom(get_timestamp(prediction_timestamp))

        # Validate and construct the optional actual
        a = None
        if actual_label is not None:
            model_type = (
                infer_model_type(actual_label) if model_type is None else model_type
            )
            _label_validation(model_type, label=convert_element(actual_label))
            a = public__pb2.Actual(
                label=_get_label(
                    value=actual_label, name="actual", model_type=model_type
                )
            )
            # Added to support latent tags on actuals.
            if tags is not None:
                converted_tags = {}
                for (k, v) in tags.items():
                    val = get_value_object(value=v, name=k)
                    if val is not None:
                        converted_tags[str(k)] = val
                tgs = public__pb2.Actual(tags=converted_tags)
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
                    raise ValueError(
                        f"feature {k} must not be named with a `_shap` suffix"
                    )
            fi = public__pb2.FeatureImportances(feature_importances=shap_values)

        if p is None and a is None and fi is None:
            raise ValueError(
                f"must provide at least one of prediction_label, actual_label, or shap_values"
            )

        rec = public__pb2.Record(
            space_key=self._space_key,
            model_id=model_id,
            prediction_id=str(prediction_id),
            prediction=p,
            actual=a,
            feature_importances=fi,
        )
        return self._post(record=rec, uri=self._uri, indexes=None)

    def _post(self, record, uri, indexes):
        resp = self._session.post(
            uri,
            headers=self._header,
            timeout=self._timeout,
            json=MessageToDict(message=record, preserving_proto_field_name=True),
        )
        if indexes is not None and len(indexes) == 2:
            resp.starting_index = indexes[0]
            resp.ending_index = indexes[1]
        return resp


def _label_validation(
    model_type: ModelTypes, label: Union[str, bool, int, float, Tuple[str, float]]
):
    if model_type == ModelTypes.BINARY:
        if not (isinstance(label, bool) or label == 0 or label == 1):
            raise TypeError(
                f"label {label} has type {type(label)}, but must be one a bool, 0 or 1 for ModelTypes.BINARY"
            )
    elif model_type == ModelTypes.NUMERIC:
        if not isinstance(label, (float, int)):
            raise TypeError(
                f"label {label} has type {type(label)}, but must be either float or int for ModelTypes.NUMERIC"
            )
        elif label is np.nan:
            raise ValueError("label for ModelTypes.NUMERIC cannot be null value")
    elif model_type == ModelTypes.CATEGORICAL:
        if not isinstance(label, str):
            raise TypeError(
                f"label {label} has type {type(label)}, but must be str for ModelTypes.CATEGORICAL"
            )
    elif model_type == ModelTypes.SCORE_CATEGORICAL:
        c = isinstance(label, str) or (
            isinstance(label, tuple)
            and isinstance(label[0], str)
            and isinstance(label[1], float)
        )
        if isinstance(label, tuple) and label[1] is np.nan:
            raise ValueError(
                f"Prediction score for ModelTypes.SCORE_CATEGORICAL cannot be null value"
            )

        if not c:
            raise TypeError(
                f"label {label} has type {type(label)}, but must be str or Tuple[str, float] for ModelTypes.SCORE_CATEGORICAL"
            )


def _get_label(
    name: str,
    value: Union[str, bool, int, float, Tuple[str, float]],
    model_type: Optional[ModelTypes],
) -> public__pb2.Label:
    if isinstance(value, public__pb2.Label):
        return value
    value = convert_element(value)
    if model_type == ModelTypes.SCORE_CATEGORICAL:
        if isinstance(value, tuple):
            return _get_score_categorical_label(value)
        else:
            sc = public__pb2.ScoreCategorical()
            sc.category.category = value
            return public__pb2.Label(score_categorical=sc)
    elif model_type == ModelTypes.BINARY:
        return public__pb2.Label(binary=value)
    elif model_type == ModelTypes.NUMERIC:
        return public__pb2.Label(numeric=value)
    elif model_type == ModelTypes.CATEGORICAL:
        return public__pb2.Label(categorical=value)
    raise TypeError(
        f"{name}_label = {value} of type {type(value)}. Must be one of str, bool, float, int, or Tuple[str, float]"
    )


def _get_score_categorical_label(value):
    sc = public__pb2.ScoreCategorical()
    if value[1] is not None:
        sc.score_category.category = value[0]
        sc.score_category.score = value[1]
    else:
        sc.category.category = value[0]

    return public__pb2.Label(score_categorical=sc)
