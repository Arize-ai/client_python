import json
import time

import pandas as pd
import numpy as np
import concurrent.futures as cf
from typing import Union, Optional, Dict, List, Tuple
from google.protobuf.json_format import MessageToDict
from requests_futures.sessions import FuturesSession

from arize import public_pb2 as public__pb2
from arize.bounded_executor import BoundedExecutor
from arize.utils.types import ModelTypes, Embedding
from arize.model import (
    TrainingRecords,
    ValidationRecords,
)
from arize.utils.utils import (
    validate_prediction_timestamps,
    bundle_records,
    convert_element,
    get_value_object,
    get_timestamp,
    is_timestamp_in_range,
    infer_model_type,
    get_bulk_records,
)

from warnings import warn

from arize.__init__ import __version__


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


def _validate_bulk_prediction(
    model_version,
    prediction_labels,
    prediction_ids,
    features,
    feature_names_overwrite,
    tags,
    tag_names_overwrite,
    prediction_timestamps,
):
    if prediction_labels.isna().values.any():
        raise ValueError("prediction labels cannot contain null value")
    if not isinstance(model_version, str):
        raise TypeError(
            f"model_version {model_version} is type {type(model_version)}, but must be a str"
        )

        # Validate prediction labels type and shape and that column length is the same as prediction ids
    if not isinstance(prediction_labels, (pd.DataFrame, pd.Series)):
        raise TypeError(
            f"prediction_labels is type {type(prediction_labels)}, but expects one of: pd.DataFrame, pd.Series"
        )
    if isinstance(prediction_labels, pd.DataFrame) and not (
        prediction_labels.shape[1] == 1 or prediction_labels.shape[1] == 2
    ):
        raise ValueError(
            f"prediction_labels contains {prediction_labels.shape[1]} columns, but can only have 1 or 2"
        )

    if isinstance(prediction_labels, pd.DataFrame) and prediction_labels.shape[1] == 2:
        if not pd.api.types.is_string_dtype(
            prediction_labels[prediction_labels.columns[0]]
        ):
            raise TypeError(
                f"Two column prediction_labels must have strings for column 0."
            )
        if not pd.api.types.is_numeric_dtype(
            prediction_labels[prediction_labels.columns[1]]
        ):
            raise TypeError(
                f"Two column prediction_labels must have numerics for column 1."
            )

    if prediction_labels.shape[0] != prediction_ids.shape[0]:
        raise ValueError(
            f"prediction_labels contains {prediction_labels.shape[0]} elements, but must have the same as "
            f"predictions_ids: {prediction_ids.shape[0]}. "
        )

        # Validate features type, shape matches prediction ids, and handle feature names overwrite
    if features is not None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                f"features is type {type(features)}, but expect type pd.DataFrame."
            )
        if features.shape[0] != prediction_ids.shape[0]:
            raise ValueError(
                f"features has {features.shape[0]} sets of features, but must match size of predictions_ids: "
                f"{prediction_ids.shape[0]}. "
            )
        if feature_names_overwrite is not None:
            if len(features.columns) != len(feature_names_overwrite):
                raise ValueError(
                    f"feature_names_overwrite has len:{len(feature_names_overwrite)}, but expects the same "
                    f"number of columns in features dataframe: {len(features.columns)}. "
                )
        else:
            for name in features.columns:
                if (
                    not isinstance(name, str)
                    and not isinstance(name, int)
                    and not isinstance(name, float)
                ):
                    raise TypeError(
                        f"features.column {name} is type {type(name)}, but expect str"
                    )

        # Validate tags type, shape matches prediction ids, and handle tag names overwrite
    if tags is not None:
        if not isinstance(tags, pd.DataFrame):
            raise TypeError(f"tags is type {type(tags)}, but expect type pd.DataFrame.")
        if tags.shape[0] != prediction_ids.shape[0]:
            raise ValueError(
                f"tags has {tags.shape[0]} sets of tags, but must match size of predictions_ids: "
                f"{prediction_ids.shape[0]}. "
            )
        if tag_names_overwrite is not None:
            if len(tags.columns) != len(tag_names_overwrite):
                raise ValueError(
                    f"tag_names_overwrite has len:{len(tag_names_overwrite)}, but expects the same "
                    f"number of columns in tags dataframe: {len(tags.columns)}. "
                )
        else:
            for name in tags.columns:
                if (
                    not isinstance(name, str)
                    and not isinstance(name, int)
                    and not isinstance(name, float)
                ):
                    raise TypeError(
                        f"tags.column {name} is type {type(name)}, but expect str"
                    )

    # Validate timestamp overwrite
    validate_prediction_timestamps(prediction_ids, prediction_timestamps)


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

    # Deprecated
    def bulk_log(
        self,
        model_id: str,
        prediction_ids: Union[pd.DataFrame, pd.Series],
        model_version: str = None,
        prediction_labels: Union[
            pd.DataFrame, pd.Series
        ] = None,  # 1xN or 2xN (for scored categorical)
        features: Optional[Union[pd.DataFrame, pd.Series]] = None,
        tags: Optional[Union[pd.DataFrame, pd.Series]] = None,
        actual_labels: Union[pd.DataFrame, pd.Series] = None,
        shap_values: Union[pd.DataFrame, pd.Series] = None,
        model_type: Optional[ModelTypes] = None,
        feature_names_overwrite: Optional[List[str]] = None,
        tag_names_overwrite: Optional[List[str]] = None,
        prediction_timestamps: Optional[Union[List[int], pd.Series]] = None,
    ) -> List[cf.Future]:
        """Logs a collection of predictions with Arize via a POST request. Returns list<:class:`Future`> object.
        :param model_id: (str) Unique identifier for a given model
        :param model_version: (str) Field used to group together a subset of predictions and actuals for a given model_id.
        :param prediction_ids: Pandas DataFrame with shape (N, 1) or Series with str valued elements. Each element corresponding to a unique string identifier for a specific prediction. These values are needed to match latent actual labels to their original prediction labels. Each element corresponds to feature values of the same index.
        :param prediction_labels: Optional Pandas DataFrame with shape (N, 1) or (N, 2) or Series. The predicted values for a given model input. Values are associates to the ids in the same index.  For a (N, 2) DataFrame column 0 is interpretted as the prediction category and column 1 is interpretted as the prediction score.
        :param features: Optional Pandas DataFrame with shape (N, 2) containing human readable and debuggable model features. DataFrames columns (df.columns) should contain feature names and must have same number of rows as prediction_ids and prediction_labels. N.B. np.nan values are stripped from the record and manifest on our platform as a missing value (not 0.0 or NaN)
        :param tags: Optional Pandas DataFrame with shape (N, 2) containing human readable and debuggable model tags. DataFrames columns (df.columns) should contain tag names and must have same number of rows as prediction_ids and prediction_labels. N.B. np.nan values are stripped from the record and manifest on our platform as a missing value (not 0.0 or NaN)
        :param actual_labels: Optional Pandas DataFrame with shape (N, 1) or Series. The actual true values for a given model input. Values are associates to the labels in the same index.
        :param shap_values: Optional Pandas DataFrame with shape (N, 1) or Series. The SHAP value sets for a set of predictions. SHAP value sets are correspond to the prediction ids with the same index.
        :param model_type: (ModelTypes) Declares what model type this prediction is for. Binary, Numeric, Categorical, Score_Categorical.
        :param feature_names_overwrite: Optional list<str> that if present will overwrite features.columns values. Must contain the same number of elements as features.columns.
        :param tag_names_overwrite: Optional list<str> that if present will overwrite tag.columns values. Must contain the same number of elements as tags.columns.
        :param prediction_timestamps: (list<int>) Optional list with same number of elements as prediction_labels field with unix epoch time in seconds to overwrite timestamp for each prediction. If None, prediction uses current timestamp.
        :rtype : list<concurrent.futures.Future>
        """
        warn(
            "The method bulk_log is deprecated and will be removed in a future version",
            DeprecationWarning,
            stacklevel=2,
        )

        # Validate model_id
        if not isinstance(model_id, str):
            raise TypeError(
                f"model_id {model_id} is type {type(model_id)}, but must be a str"
            )

        # Validate prediction_ids
        if not isinstance(prediction_ids, (pd.DataFrame, pd.Series)):
            raise TypeError(
                f"prediction_ids is type {type(prediction_ids)}, but expect one of: pd.DataFrame, pd.Series"
            )

        if prediction_labels is not None:
            _validate_bulk_prediction(
                model_version,
                prediction_labels,
                prediction_ids,
                features,
                feature_names_overwrite,
                tags,
                tag_names_overwrite,
                prediction_timestamps,
            )
            model_type = (
                infer_model_type(prediction_labels.iloc[0])
                if model_type is None
                else model_type
            )

        if actual_labels is not None:
            if not isinstance(actual_labels, (pd.DataFrame, pd.Series)):
                raise TypeError(
                    f"actual_labels is type: {type(actual_labels)}, but expects one of: pd.DataFrame, pd.Series"
                )
            if actual_labels.shape[0] != prediction_ids.shape[0]:
                raise ValueError(
                    f"actual_labels contains {actual_labels.shape[0]} elements, but must have the same as "
                    f"predictions_ids: {prediction_ids.shape[0]}. "
                )
            # Set model type if not yet set
            model_type = (
                infer_model_type(actual_labels.iloc[0])
                if model_type is None
                else model_type
            )

        if shap_values is not None:
            if not isinstance(shap_values, pd.DataFrame):
                raise TypeError(
                    f"shap_values is type {type(shap_values)}, but expect type pd.DataFrame."
                )
            if shap_values.shape[0] != prediction_ids.shape[0]:
                raise ValueError(
                    f"shap_values has {shap_values.shape[0]} sets of values, but must match size of "
                    f"predictions_ids: {shap_values.shape[0]}. "
                )
            if isinstance(shap_values.columns, pd.core.indexes.numeric.NumericIndex):
                raise TypeError(
                    f"shap_values.columns is of type {type(shap_values.columns)}, but expect elements to be str."
                )
            for name in shap_values.columns:
                if not isinstance(name, str):
                    raise TypeError(
                        f"shap_values.column {name} is type {type(name)}, but expect str"
                    )
                if isinstance(name, str) and name.endswith("_shap"):
                    raise ValueError(
                        f"feature {name} must not be named with a `_shap` suffix"
                    )

        prediction_ids = prediction_ids.to_numpy()
        prediction_labels = (
            prediction_labels.to_numpy() if prediction_labels is not None else None
        )
        prediction_timestamps = (
            prediction_timestamps.tolist()
            if isinstance(prediction_timestamps, pd.Series)
            else prediction_timestamps
        )
        if features is not None:
            feature_names = feature_names_overwrite or features.columns
            for name in feature_names:
                if isinstance(name, str) and name.endswith("_shap"):
                    raise ValueError(
                        f"feature {name} must not be named with a `_shap` suffix"
                    )
            features = features.to_numpy()
        if tags is not None:
            tag_names = tag_names_overwrite or tags.columns
            for name in tag_names:
                if isinstance(name, str) and name.endswith("_shap"):
                    raise ValueError(
                        f"tag {name} must not be named with a `_shap` suffix"
                    )
            tags = tags.to_numpy()
        actual_labels = actual_labels.to_numpy() if actual_labels is not None else None
        shap_columns = shap_values.columns if shap_values is not None else None
        shap_values = shap_values.to_numpy() if shap_values is not None else None

        records = []
        for row, v in enumerate(prediction_ids):
            pred_id = (
                v
                if (isinstance(v, str) or isinstance(v, int) or isinstance(v, float))
                else v[0]
            )
            p = None
            if prediction_labels is not None:
                # if there is more than 1 dimension, and the second dimension size is 2 - TODO instead just guarantee shape is always (X,Y) instead of sometimes (X,)
                if (
                    len(prediction_labels.shape) == 2
                    and prediction_labels.shape[1] == 2
                ):
                    label = _get_label(
                        value=(prediction_labels[row][0], prediction_labels[row][1]),
                        name="prediction",
                        model_type=model_type,
                    )
                else:
                    label = _get_label(
                        value=prediction_labels[row],
                        name="prediction",
                        model_type=model_type,
                    )
                p = public__pb2.Prediction(label=label)
                if features is not None:
                    converted_feats = {}
                    for column, name in enumerate(feature_names):
                        val = get_value_object(value=features[row][column], name=name)
                        if val is not None:
                            converted_feats[str(name)] = val
                    feats = public__pb2.Prediction(features=converted_feats)
                    p.MergeFrom(feats)
                if tags is not None:
                    converted_tags = {}
                    for column, name in enumerate(tag_names):
                        val = get_value_object(value=tags[row][column], name=name)
                        if val is not None:
                            converted_tags[str(name)] = val
                    tgs = public__pb2.Prediction(tags=converted_tags)
                    p.MergeFrom(tgs)
                if prediction_timestamps is not None:
                    p.timestamp.MergeFrom(get_timestamp(prediction_timestamps[row]))

            a = None
            if actual_labels is not None:
                a = public__pb2.Actual(
                    label=_get_label(
                        value=actual_labels[row], name="actual", model_type=model_type
                    )
                )

            fi = None
            if shap_values is not None:
                converted_fi = {
                    name: shap_values[row][column]
                    for column, name in enumerate(shap_columns)
                }
                fi = public__pb2.FeatureImportances(feature_importances=converted_fi)

            rec = public__pb2.Record(
                prediction_id=str(pred_id),
                prediction=p,
                actual=a,
                feature_importances=fi,
            )
            records.append(rec)
        brs = get_bulk_records(
            self._space_key, model_id, model_version, bundle_records(records)
        )
        return self._post_bulk(records=brs, uri=self._bulk_url)

    # Deprecated
    def log_validation_records(
        self,
        model_id: str,
        model_version: str,
        batch_id: str,
        prediction_labels: Union[pd.DataFrame, pd.Series],
        actual_labels: Union[pd.DataFrame, pd.Series],
        prediction_scores: Optional[Union[pd.DataFrame, pd.Series]] = None,
        prediction_ids: Optional[Union[pd.DataFrame, pd.Series]] = None,
        model_type: Optional[ModelTypes] = None,
        features: Optional[Union[pd.DataFrame, pd.Series]] = None,
        tags: Optional[Union[pd.DataFrame, pd.Series]] = None,
        prediction_timestamps: Optional[Union[List[int], pd.Series]] = None,
    ) -> List[cf.Future]:
        """Logs a set of validation records to Arize. Returns :class:`Future` object.
        :param model_id: (str) Unique identifier for a given model.
        :param model_type: (ModelTypes) Declares what model type these records are for. Binary, Numeric, Categorical, Score_Categorical.
        :param model_version: (str) Unique identifier used to group together a subset of records for a given model_id.
        :param batch_id: (str) Unique identifier used to group together a subset of validation records for a given model_id and model_version - akin to a validation set.
        :param prediction_labels: 1-D Pandas DataFrame or Series. The predicted values for a given model input.
        :param actual_labels: 1-D Pandas DataFrame or Series. The actual true values for a given model input.
        :param prediction_scores: 1-D Pandas DataFrame or Series. The predicted scores for the corresponding predicted_label of classification model. If present, elements in prediction_labels must be of type str. Values are associates to the labels in the same index.
        :param prediction_ids: 1-D Pandas DataFrame or Series. The prediction IDs for the corresponding predicted_label of a classification_model. If present, elements in prediction_labels must be of type str. Values are associates to the labels in the same index.
        :param features: Optional 2-D Pandas DataFrame containing human readable and debuggable model features. DataFrames columns (df.columns) should contain feature names and must have same number of rows as actual_labels and prediction_labels. N.B. np.nan values are stripped from the record and manifest on our platform as a missing value (not 0.0 or NaN)
        :param tags: Optional 2-D Pandas DataFrame containing human readable and debuggable model tags. DataFrames columns (df.columns) should contain tag names and must have same number of rows as actual_labels and prediction_labels. N.B. np.nan values are stripped from the record and manifest on our platform as a missing value (not 0.0 or NaN)
        :param prediction_timestamps: (list<int>) Optional list with same number of elements as prediction_labels field with unix epoch time in seconds to overwrite timestamp for each prediction. If None, prediction uses current timestamp.
        :rtype : list<concurrent.futures.Future>
        """
        warn(
            "The method log_validation_records is deprecated and will be removed in a future version",
            DeprecationWarning,
            stacklevel=2,
        )
        rec = ValidationRecords(
            space_key=self._space_key,
            model_id=model_id,
            model_type=model_type,
            model_version=model_version,
            batch_id=batch_id,
            features=features,
            tags=tags,
            prediction_labels=prediction_labels,
            actual_labels=actual_labels,
            prediction_scores=prediction_scores,
            prediction_ids=prediction_ids,
            prediction_timestamps=prediction_timestamps,
        )
        rec.validate_inputs()
        return self._post_preprod(records=rec.build_proto())

    # Deprecated
    def log_training_records(
        self,
        model_id: str,
        model_version: str,
        prediction_labels: Union[pd.DataFrame, pd.Series],
        actual_labels: Union[pd.DataFrame, pd.Series],
        prediction_scores: Optional[Union[pd.DataFrame, pd.Series]] = None,
        model_type: Optional[ModelTypes] = None,
        features: Optional[Union[pd.DataFrame, pd.Series]] = None,
        tags: Optional[Union[pd.DataFrame, pd.Series]] = None,
    ) -> List[cf.Future]:
        """Logs a stream of training records to Arize. Returns :class:`Future` object.
        :param model_id: (str) Unique identifier for a given model.
        :param model_version: (str) Unique identifier used to group together a subset of records for a given model_id.
        :param model_type: (ModelTypes) Declares what model type these records are for. Binary, Numeric, Categorical, Score_Categorical.
        :param prediction_labels: 1-D Pandas DataFrame or Series. The predicted values for a given model input.
        :param actual_labels: 1-D Pandas DataFrame or Series. The actual true values for a given model input.
        :param prediction_scores: 1-D Pandas DataFrame or Series. The predicted scores for the corresponding predicted_label of classification model. If present, elements in prediction_labels must be of type str. Values are associates to the labels in the same index.
        :param features: Optional 2-D Pandas DataFrame containing human readable and debuggable model features. DataFrames columns (df.columns) should contain feature names and must have same number of rows as actual_labels and prediction_labels. N.B. np.nan values are stripped from the record and manifest on our platform as a missing value (not 0.0 or NaN)
        :param tags: Optional 2-D Pandas DataFrame containing human readable and debuggable model tags. DataFrames columns (df.columns) should contain tag names and must have same number of rows as actual_labels and prediction_labels. N.B. np.nan values are stripped from the record and manifest on our platform as a missing value (not 0.0 or NaN)
        :rtype : list<concurrent.futures.Future>
        """
        warn(
            "The method log_training_records is deprecated and will be removed in a future version",
            DeprecationWarning,
            stacklevel=2,
        )
        rec = TrainingRecords(
            space_key=self._space_key,
            model_id=model_id,
            model_type=model_type,
            model_version=model_version,
            features=features,
            tags=tags,
            prediction_labels=prediction_labels,
            prediction_scores=prediction_scores,
            actual_labels=actual_labels,
        )
        rec.validate_inputs()
        return self._post_preprod(records=rec.build_proto())

    # Deprecated
    def log_prediction(
        self,
        model_id: str,
        model_version: str,
        prediction_id: str,
        prediction_label: Union[str, bool, int, float],
        prediction_score: Optional[float] = None,
        features: Optional[Dict[str, Union[str, bool, float, int]]] = None,
        tags: Optional[Dict[str, Union[str, bool, float, int]]] = None,
        model_type: Optional[ModelTypes] = None,
        time_overwrite: Optional[int] = None,
    ) -> cf.Future:
        """Logs a prediction to Arize via a POST request. Returns :class:`Future` object.
        :param model_id: (str) Unique identifier for a given model
        :param model_version: (str) Field used to group together a subset of predictions and actuals for a given model_id.
        :param prediction_id: (str) Unique string identifier for a specific prediction. This value is used to match a prediction to an actual label in the Arize platform.
        :param prediction_label: (one of bool, str, float, int) The predicted value for a given model input.
        :param prediction_score: (float) Optional predicted score for the predicted_label of classification model. If present, the prediction_label must be of type str.
        :param features: (str, <value>) Optional dictionary containing human readable and debuggable model features. Keys must be str, values one of str, bool, float, long.
        :param tags: (str, <value>) Optional dictionary containing human readable and debuggable model tags. Keys must be str, values one of str, bool, float, long.
        :param model_type: (ModelTypes) Declares what model type this prediction is for. Binary, Numeric, Categorical, Score_Categorical.
        :param time_overwrite: (int) Optional field with unix epoch time in seconds to overwrite timestamp for prediction. If None, prediction uses current timestamp.
        :rtype : concurrent.futures.Future
        """
        warn(
            "The method log_prediction is deprecated and will be removed in a future version",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.log(
            model_id=model_id,
            prediction_id=prediction_id,
            model_version=model_version,
            prediction_label=prediction_label
            if prediction_score is None
            else (prediction_label, prediction_score),
            features=features,
            model_type=model_type,
            prediction_timestamp=time_overwrite,
        )

    # Deprecated
    def log_actual(
        self,
        model_id: str,
        prediction_id: str,
        actual_label: Union[str, bool, int, float],
        model_type: Optional[ModelTypes] = None,
    ) -> cf.Future:
        """Logs an actual to Arize via a POST request. Returns :class:`Future` object.
        :param model_id: (str) Unique identifier for a given model
        :param prediction_id: (str) Unique string identifier for a specific prediction. This value is used to match a prediction to an actual label in the Arize platform.
        :param actual_label: (one of bool, str, float, int) The actual true value for a given model input. This actual will be matched to the prediction with the same prediction_id as the one in this call.
        :param model_type: (ModelTypes) Declares what model type this prediction is for. Binary, Numeric, Categorical, Score_Categorical.
        :rtype : concurrent.futures.Future
        """
        warn(
            "The method log_actual is deprecated and will be removed in a future version",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.log(
            model_id=model_id,
            model_type=model_type,
            prediction_id=prediction_id,
            actual_label=actual_label,
        )

    # Deprecated
    def log_shap_values(
        self,
        model_id: str,
        prediction_id: str,
        shap_values: Dict[str, float],
    ) -> cf.Future:
        """Logs SHAP feature importance values for a given prediction to Arize via a POST request. Returns :class:`Future` object.
        :param model_id: (str) Unique identifier for a given model.
        :param prediction_id: (str) Unique string identifier for a specific prediction. This value is used to match a prediction to the SHAP values supplied in this request in the Arize platform.
        :param shap_values: (str, float) Dictionary containing human readable and debuggable model features keys, along with SHAP feature importance values. Keys must be str, while values must be float.
        :rtype : concurrent.futures.Future
        """
        warn(
            "The method log_shap_values is deprecated and will be removed in a future version",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.log(
            prediction_id=prediction_id,
            model_id=model_id,
            shap_values=shap_values,
        )

    # Deprecated
    def log_bulk_shap_values(
        self,
        model_id: str,
        prediction_ids: Union[pd.DataFrame, pd.Series],
        shap_values: Union[pd.DataFrame, pd.Series],
    ) -> List[cf.Future]:
        """Logs a collection of SHAP feature importance value sets with Arize via a POST request. Returns list<:class:`Future`> object.
        :param model_id: (str) Unique identifier for a given model
        :param prediction_ids: 1-D Pandas DataFrame or Series with string elements. Each element corresponding to a unique string identifier for a specific prediction. Each element corresponds to the SHAP values of the same index.
        :param shap_values: 1-D Pandas DataFrame or Series. The SHAP value sets for a set of predictions. SHAP value sets are correspond to the prediction ids with the same index.
        :rtype : list<concurrent.futures.Future>
        """
        warn(
            "The method log_bulk_shap_values is deprecated and will be removed in a future version",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.bulk_log(
            model_id=model_id, prediction_ids=prediction_ids, shap_values=shap_values
        )

    # Deprecated
    def log_bulk_predictions(
        self,
        model_id: str,
        model_version: str,
        prediction_ids: Union[pd.DataFrame, pd.Series],
        prediction_labels: Union[pd.DataFrame, pd.Series],
        prediction_scores: Optional[Union[pd.DataFrame, pd.Series]] = None,
        features: Optional[Union[pd.DataFrame, pd.Series]] = None,
        tags: Optional[Union[pd.DataFrame, pd.Series]] = None,
        model_type: Optional[ModelTypes] = None,
        feature_names_overwrite: Optional[List[str]] = None,
        tag_names_overwrite: Optional[List[str]] = None,
        time_overwrite: Optional[List[int]] = None,
    ) -> List[cf.Future]:
        """Logs a collection of predictions with Arize via a POST request. Returns list<:class:`Future`> object.
        :param model_id: (str) Unique identifier for a given model
        :param model_type: (ModelTypes) Declares what model type this prediction is for. Binary, Numeric, Categorical, Score_Categorical.
        :param model_version: (str) Field used to group together a subset of predictions and actuals for a given model_id.
        :param prediction_ids: 1-D Pandas DataFrame or Series with string elements. Each element corresponding to a unique string identifier for a specific prediction. These values are needed to match latent actual labels to their original prediction labels. Each element corresponds to feature values of the same index.
        :param prediction_labels: 1-D Pandas DataFrame or Series. The predicted values for a given model input. Values are associates to the ids in the same index.
        :param prediction_scores: 1-D Pandas DataFrame or Series. The predicted scores for the corresponding predicted_label of classification model. If present, elements in prediction_labels must be of type str. Values are associates to the labels in the same index.
        :param features: Optional 2-D Pandas DataFrame containing human readable and debuggable model features. DataFrames columns (df.columns) should contain feature names and must have same number of rows as prediction_ids and prediction_labels. N.B. np.nan values are stripped from the record and manifest on our platform as a missing value (not 0.0 or NaN)
        :param feature_names_overwrite: Optional list<str> that if present will overwrite features.columns values. Must contain the same number of elements as features.columns.
        :param tags: Optional 2-D Pandas DataFrame containing human readable and debuggable model tags. DataFrames columns (df.columns) should contain tag names and must have same number of rows as prediction_ids and prediction_labels. N.B. np.nan values are stripped from the record and manifest on our platform as a missing value (not 0.0 or NaN)
        :param tag_names_overwrite: Optional list<str> that if present will overwrite tags.columns values. Must contain the same number of elements as tags.columns.
        :param time_overwrite: (list<int>) Optional list with same number of elements as prediction_labels field with unix epoch time in seconds to overwrite timestamp for each prediction. If None, prediction uses current timestamp.
        :rtype : list<concurrent.futures.Future>
        """
        warn(
            "The method log_bulk_predictions is deprecated and will be removed in a future version",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.bulk_log(
            model_id=model_id,
            model_type=model_type,
            model_version=model_version,
            prediction_ids=prediction_ids,
            prediction_labels=pd.concat([prediction_labels, prediction_scores], axis=1),
            features=features,
            feature_names_overwrite=feature_names_overwrite,
            tags=tags,
            tag_names_overwrite=tag_names_overwrite,
            prediction_timestamps=time_overwrite,
        )

    # Deprecated
    def log_bulk_actuals(
        self,
        model_id: str,
        prediction_ids: Union[pd.DataFrame, pd.Series],
        actual_labels: Union[pd.DataFrame, pd.Series],
        model_type: Optional[ModelTypes] = None,
    ) -> List[cf.Future]:
        """Logs a collection of actuals with Arize via a POST request. Returns list<:class:`Future`> object.
        :param model_id: (str) Unique identifier for a given model
        :param model_type: (ModelTypes) Declares what model type this prediction is for. Binary, Numeric, Categorical, Score_Categorical.
        :param prediction_ids: 1-D Pandas DataFrame or Series with string elements. Each element corresponding to a unique string identifier for a specific prediction. These values are needed to match latent actual labels to their original prediction labels. Each element corresponds to feature values of the same index.
        :param actual_labels: 1-D Pandas DataFrame or Series. The actual true values for a given model input. Values are associates to the labels in the same index.
        :rtype : list<concurrent.futures.Future>
        """
        warn(
            "The method log_bulk_actuals is deprecated and will be removed in a future version",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.bulk_log(
            model_id=model_id,
            model_type=model_type,
            prediction_ids=prediction_ids,
            actual_labels=actual_labels,
        )

    def _post_bulk(self, records, uri):
        return [self._post(r, uri, k) for k, r in records.items()]

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

    def _post_preprod(self, records):
        futures = []
        for k, recs in records.items():
            futures.append(
                self._session.post(
                    self._stream_uri,
                    headers=self._header,
                    timeout=self._timeout,
                    data="\n".join(
                        json.dumps(
                            MessageToDict(message=d, preserving_proto_field_name=True)
                        )
                        for d in recs
                    ),
                )
            )
        return futures
