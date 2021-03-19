from abc import ABC, abstractmethod

import pandas as pd
from typing import Optional, Union, Dict

from arize import public_pb2 as public__pb2
from arize.types import ModelTypes
from arize.utils import (
    bundle_records,
    convert_element,
    get_value_object,
    get_timestamp,
    infer_model_type,
    get_bulk_records,
)


class BaseRecord(ABC):
    def __init__(
        self,
        organization_key: str,
        model_id: str,
        model_type: Optional[ModelTypes] = None,
    ):
        self.organization_key = organization_key
        self.model_id = model_id
        self.model_type = model_type

    @abstractmethod
    def validate_inputs(self):
        pass

    @abstractmethod
    def build_proto(self):
        pass

    def _base_validation(self):
        if not isinstance(self.organization_key, str):
            raise TypeError(
                f"organization_key {self.organization_key} is type {type(self.organization_key)}, but must be a str"
            )
        if not isinstance(self.model_id, str):
            raise TypeError(
                f"model_id {self.model_id} is type {type(self.model_id)}, but must be a str"
            )

    def _label_validation(self, label):
        if self.model_type == ModelTypes.BINARY:
            if not (isinstance(label, bool) or label == 0 or label == 1):
                raise TypeError(
                    f"label {label} has type {type(label)}, but must be one a bool, 0 or 1 for ModelTypes.BINARY"
                )
        elif self.model_type == ModelTypes.NUMERIC:
            if not isinstance(label, (float, int)):
                raise TypeError(
                    f"label {label} has type {type(label)}, but must be either float or int for ModelTypes.NUMERIC"
                )
        elif self.model_type == ModelTypes.CATEGORICAL:
            if not isinstance(label, str):
                raise TypeError(
                    f"label {label} has type {type(label)}, but must be str for ModelTypes.CATEGORICAL"
                )
        elif self.model_type == ModelTypes.SCORE_CATEGORICAL:
            if not isinstance(label, str):
                raise TypeError(
                    f"label {label} has type {type(label)}, but must be str for ModelTypes.SCORE_CATEGORICAL"
                )

    def _get_label(
        self, name: str, value, score: Optional[float] = None
    ) -> public__pb2.Label:
        if isinstance(value, public__pb2.Label):
            return value
        val = convert_element(value)
        if self.model_type == ModelTypes.SCORE_CATEGORICAL:
            return public__pb2.Label(
                score_categorical=public__pb2.ScoreCategorical(
                    categorical=val, score=convert_element(score)
                )
            )
        elif self.model_type == ModelTypes.BINARY:
            return public__pb2.Label(binary=val)
        elif self.model_type == ModelTypes.NUMERIC:
            return public__pb2.Label(numeric=val)
        elif self.model_type == ModelTypes.CATEGORICAL:
            return public__pb2.Label(categorical=val)
        raise TypeError(
            f"{name}_label = {value} of type {type(value)}. Must be one of str, bool, float, or int"
        )


class Prediction(BaseRecord):
    def __init__(
        self,
        organization_key: str,
        model_id: str,
        model_version: str,
        prediction_id: str,
        prediction_label: Union[str, float, int, bool],
        prediction_score: Optional[float] = None,
        features: Optional[Dict[str, Union[str, float, int, bool]]] = None,
        time_overwrite: Optional[int] = None,
        model_type: Optional[ModelTypes] = None,
    ):
        if model_type is None:
            model_type = infer_model_type(prediction_label)
        super().__init__(
            organization_key=organization_key, model_id=model_id, model_type=model_type
        )
        self.model_version = model_version
        self.prediction_id = prediction_id
        self.prediction_label = prediction_label
        self.prediction_score = prediction_score
        self.features = features
        self.time_overwrite = time_overwrite

    def validate_inputs(self):
        self._base_validation()
        if not isinstance(self.model_version, str):
            raise TypeError(
                f"model_version {self.model_version} is type {type(self.model_version)}, but must be a str"
            )
        if self.features is not None and bool(self.features):
            for k, v in self.features.items():
                if not isinstance(convert_element(v), (str, bool, float, int)):
                    raise TypeError(
                        f"feature {k} with value {v} is type {type(v)}, but expected one of: str, bool, float, int"
                    )
        if self.time_overwrite is not None and not isinstance(self.time_overwrite, int):
            raise TypeError(
                f"time_overwrite {self.time_overwrite} is type {type(self.time_overwrite)} but expected int"
            )
        self._label_validation(label=convert_element(self.prediction_label))
        if (self.model_type == ModelTypes.SCORE_CATEGORICAL) and (
            not isinstance(convert_element(self.prediction_score), float)
        ):
            raise TypeError(
                f"ModelTypes.SCORE_CATEGORICAL requires a prediction_score (float) but got prediction_score ({type(self.prediction_score)})"
            )

    def build_proto(self):
        p = public__pb2.Prediction(
            label=self._get_label(
                value=self.prediction_label,
                score=self.prediction_score,
                name="prediction",
            )
        )
        if self.features is not None:
            feats = public__pb2.Prediction(
                features={
                    k: get_value_object(value=v, name=k)
                    for (k, v) in self.features.items()
                }
            )
            p.MergeFrom(feats)
        if self.model_version is not None:
            p.model_version = self.model_version
        if self.time_overwrite is not None:
            p.timestamp.MergeFrom(get_timestamp(self.time_overwrite))
        return public__pb2.Record(
            organization_key=self.organization_key,
            model_id=self.model_id,
            prediction_id=self.prediction_id,
            prediction=p,
        )


class Actual(BaseRecord):
    def __init__(
        self,
        organization_key: str,
        model_id: str,
        prediction_id: str,
        actual_label: Union[str, float, int, bool],
        model_type: Optional[ModelTypes] = None,
    ):
        if model_type is None:
            model_type = infer_model_type(actual_label)
        super().__init__(
            organization_key=organization_key, model_id=model_id, model_type=model_type
        )
        self.prediction_id = prediction_id
        self.actual_label = actual_label

    def validate_inputs(self) -> None:
        self._base_validation()
        self._label_validation(convert_element(self.actual_label))

    def build_proto(self) -> public__pb2.Record:
        a = public__pb2.Actual(
            label=self._get_label(value=self.actual_label, name="actual")
        )
        return public__pb2.Record(
            organization_key=self.organization_key,
            model_id=self.model_id,
            prediction_id=self.prediction_id,
            actual=a,
        )


class FeatureImportances:
    def __init__(
        self,
        organization_key: str,
        model_id: str,
        prediction_id: str,
        feature_importances: Dict[str, float],
    ):
        self.organization_key = organization_key
        self.model_id = model_id
        self.prediction_id = prediction_id
        self.feature_importances = feature_importances

    def validate_inputs(self):
        if not isinstance(self.organization_key, str):
            raise TypeError(
                f"organization_key {self.organization_key} is type {type(self.organization_key)}, but must be a str"
            )
        if not isinstance(self.model_id, str):
            raise TypeError(
                f"model_id {self.model_id} is type {type(self.model_id)}, but must be a str"
            )
        if self.feature_importances is not None and bool(self.feature_importances):
            for k, v in self.feature_importances.items():
                if not isinstance(convert_element(v), float):
                    raise TypeError(
                        f"feature {k} with value {v} is type {type(v)}, but expected one of: float"
                    )
        elif self.feature_importances is None or len(self.feature_importances) == 0:
            raise ValueError(f"at least one feature importance value must be provided")

    def build_proto(self) -> public__pb2.Record:
        fi = public__pb2.FeatureImportances()

        if self.feature_importances is not None:
            feat_importances = public__pb2.FeatureImportances(
                feature_importances=self.feature_importances
            )
            fi.MergeFrom(feat_importances)
        return public__pb2.Record(
            organization_key=self.organization_key,
            model_id=self.model_id,
            prediction_id=self.prediction_id,
            feature_importances=fi,
        )


class BaseBulkRecord(BaseRecord, ABC):
    prediction_labels, actual_labels, time_overwrite, features, prediction_scores = (
        None,
        None,
        None,
        None,
        None,
    )

    def __init__(
        self,
        organization_key: str,
        model_id: str,
        prediction_ids: Union[pd.DataFrame, pd.Series],
        model_type: Optional[ModelTypes] = None,
    ):
        super().__init__(
            organization_key=organization_key, model_id=model_id, model_type=model_type
        )
        self.prediction_ids = prediction_ids

    def _base_bulk_validation(self):
        self._base_validation()
        if not isinstance(self.prediction_ids, (pd.DataFrame, pd.Series)):
            raise TypeError(
                f"prediction_ids is type {type(self.prediction_ids)}, but expect one of: pd.DataFrame, pd.Series"
            )

    def _normalize_inputs(self):
        """Converts inputs from DataFrames, Series, lists to numpy arrays or lists for consistent iterations
        downstream."""
        self.prediction_ids = self.prediction_ids.to_numpy()
        if isinstance(self.prediction_labels, (pd.DataFrame, pd.Series)):
            self.prediction_labels = self.prediction_labels.to_numpy()
        if isinstance(self.actual_labels, (pd.DataFrame, pd.Series)):
            self.actual_labels = self.actual_labels.to_numpy()
        if isinstance(self.time_overwrite, pd.Series):
            self.time_overwrite = self.time_overwrite.tolist()
        if isinstance(self.features, pd.DataFrame):
            self.feature_names = self.feature_names_overwrite or self.features.columns
            self.features = self.features.to_numpy()
        if isinstance(self.prediction_scores, (pd.DataFrame, pd.Series)):
            self.prediction_scores = self.prediction_scores.to_numpy()


class BulkPrediction(BaseBulkRecord):
    def __init__(
        self,
        organization_key: str,
        model_id: str,
        model_type: ModelTypes,
        model_version: str,
        prediction_ids,
        prediction_labels,
        features,
        feature_names_overwrite,
        time_overwrite,
        prediction_scores,
    ):
        super().__init__(
            organization_key=organization_key,
            model_id=model_id,
            prediction_ids=prediction_ids,
            model_type=model_type,
        )
        self.model_version = model_version
        self.prediction_labels = prediction_labels
        self.prediction_scores = prediction_scores
        self.features = features
        self.feature_names_overwrite = feature_names_overwrite
        self.time_overwrite = time_overwrite

    def validate_inputs(self):
        self._base_bulk_validation()
        if not isinstance(self.prediction_labels, (pd.DataFrame, pd.Series)):
            raise TypeError(
                f"prediction_labels is type {type(self.prediction_labels)}, but expects one of: pd.DataFrame, pd.Series"
            )
        if self.prediction_labels.shape[0] != self.prediction_ids.shape[0]:
            raise ValueError(
                f"prediction_labels contains {self.prediction_labels.shape[0]} elements, but must have the same as "
                f"predictions_ids: {self.prediction_ids.shape[0]}. "
            )
        self._validate_scored_categorical()
        self._validate_features()
        self._validate_time_overwrite()

    def build_proto(self):
        self._normalize_inputs()
        if self.model_type is None:
            self.model_type = infer_model_type(self.prediction_labels[0])
        records = []
        for row, v in enumerate(self.prediction_ids):
            pred_id = v if isinstance(v, str) else v[0]
            if not isinstance(pred_id, (str, bytes)):
                raise TypeError(
                    f"prediction_id {pred_id} is type {type(pred_id)}, but expected one of: str, bytes"
                )
            if self.prediction_scores is not None:
                label = self._get_label(
                    value=self.prediction_labels[row],
                    name="prediction",
                    score=self.prediction_scores[row],
                )
            else:
                label = self._get_label(
                    value=self.prediction_labels[row], name="prediction"
                )
            p = public__pb2.Prediction(label=label)
            if self.features is not None:
                converted_feats = {}
                for column, name in enumerate(self.feature_names):
                    val = get_value_object(value=self.features[row][column], name=name)
                    if val is not None:
                        converted_feats[name] = val
                feats = public__pb2.Prediction(features=converted_feats)
                p.MergeFrom(feats)
            if self.time_overwrite is not None:
                p.timestamp.MergeFrom(get_timestamp(self.time_overwrite[row]))

            records.append(public__pb2.Record(prediction_id=pred_id, prediction=p))
        return get_bulk_records(
            organization_key=self.organization_key,
            model_id=self.model_id,
            model_version=self.model_version,
            records=bundle_records(records),
        )

    def _validate_features(self):
        if self.features is None:
            return
        if not isinstance(self.features, pd.DataFrame):
            raise TypeError(
                f"features is type {type(self.features)}, but expect type pd.DataFrame."
            )
        if self.features.shape[0] != self.prediction_ids.shape[0]:
            raise ValueError(
                f"features has {self.features.shape[0]} sets of features, but must match size of predictions_ids: "
                f"{self.prediction_ids.shape[0]}. "
            )
        if self.feature_names_overwrite is not None:
            if len(self.features.columns) != len(self.feature_names_overwrite):
                raise ValueError(
                    f"feature_names_overwrite has len:{len(self.feature_names_overwrite)}, but expects the same "
                    f"number of columns in features dataframe: {len(self.features.columns)}. "
                )
        else:
            if isinstance(self.features.columns, pd.core.indexes.numeric.NumericIndex):
                raise TypeError(
                    f"features.columns is of type {type(self.features.columns)}, but expect elements to be str. "
                    f"Alternatively, feature_names_overwrite must be present. "
                )
            for name in self.features.columns:
                if not isinstance(name, str):
                    raise TypeError(
                        f"features.column {name} is type {type(name)}, but expect str"
                    )

    def _validate_time_overwrite(self):
        if self.time_overwrite is None:
            return
        expected_count = self.prediction_ids.shape[0]
        if isinstance(self.time_overwrite, pd.Series):
            if self.time_overwrite.shape[0] != expected_count:
                raise ValueError(
                    f"time_overwrite has {self.time_overwrite.shape[0]} elements, but must have same number of "
                    f"elements as prediction_ids: {expected_count}. "
                )
        elif isinstance(self.time_overwrite, list):
            if len(self.time_overwrite) != expected_count:
                raise ValueError(
                    f"time_overwrite has length {len(self.time_overwrite)} but must have same number of elements as "
                    f"prediction_ids: {expected_count}. "
                )
        else:
            raise TypeError(
                f"time_overwrite is type {type(self.time_overwrite)}, but expected one of: pd.Series, list<int>"
            )

    def _validate_scored_categorical(self):
        if self.prediction_scores is None:
            return
        if not isinstance(self.prediction_scores, (pd.DataFrame, pd.Series)):
            raise TypeError(
                f"prediction_scores is type {type(self.prediction_scores)}, but expects one of: pd.DataFrame, pd.Series"
            )
        if self.prediction_scores.shape[0] != self.prediction_ids.shape[0]:
            raise ValueError(
                f"prediction_scores len({self.prediction_scores.shape[0]}) must match size of prediction_ids: "
                f"{self.prediction_ids.shape[0]}. "
            )
        if self.prediction_scores.shape[0] != self.prediction_labels.shape[0]:
            raise ValueError(
                f"prediction_scores len({self.prediction_scores.shape[0]}) must match size of prediction_labels: "
                f"{self.prediction_labels.shape[0]}. "
            )
        if not pd.api.types.is_numeric_dtype(self.prediction_scores):
            raise TypeError(
                f"prediction_scores elements is type {type(self.prediction_scores.dtype)}, but expect type numeric."
            )
        if not pd.api.types.is_string_dtype(self.prediction_labels):
            raise TypeError(
                f"prediction_labels elements is type {type(self.prediction_labels.dtype)}, but expect type string."
            )


class BulkActual(BaseBulkRecord):
    def __init__(
        self,
        organization_key: str,
        model_id: str,
        model_type: Optional[ModelTypes],
        prediction_ids: Union[pd.DataFrame, pd.Series],
        actual_labels: Union[pd.DataFrame, pd.Series],
    ):
        super().__init__(
            organization_key=organization_key,
            model_id=model_id,
            model_type=model_type,
            prediction_ids=prediction_ids,
        )
        self.actual_labels = actual_labels

    def validate_inputs(self):
        self._base_bulk_validation()
        if not isinstance(self.actual_labels, (pd.DataFrame, pd.Series)):
            raise TypeError(
                f"actual_labels is type: {type(self.actual_labels)}, but expects one of: pd.DataFrame, pd.Series"
            )
        if self.actual_labels.shape[0] != self.prediction_ids.shape[0]:
            raise ValueError(
                f"actual_labels contains {self.actual_labels.shape[0]} elements, but must have the same as "
                f"predictions_ids: {self.prediction_ids.shape[0]}. "
            )

    def build_proto(self):
        self._normalize_inputs()
        if self.model_type is None:
            self.model_type = infer_model_type(self.actual_labels[0])
        records = []
        for i, v in enumerate(self.prediction_ids):
            pred_id = v if isinstance(v, str) else v[0]
            if not isinstance(pred_id, (str, bytes)):
                raise TypeError(
                    f"prediction_id {pred_id} is type {type(pred_id)}, but expected one of: str, bytes"
                )
            a = public__pb2.Actual(
                label=self._get_label(value=self.actual_labels[i], name="actual")
            )
            records.append(public__pb2.Record(prediction_id=pred_id, actual=a))
        return get_bulk_records(
            organization_key=self.organization_key,
            model_id=self.model_id,
            model_version=None,
            records=bundle_records(records),
        )


class BulkFeatureImportances:
    def __init__(
        self,
        organization_key: str,
        model_id: str,
        prediction_ids: Union[pd.DataFrame, pd.Series],
        feature_importances,
    ):
        self.organization_key = organization_key
        self.model_id = model_id
        self.prediction_ids = prediction_ids
        self.feature_importances = feature_importances

    def validate_inputs(self):
        if not isinstance(self.organization_key, str):
            raise TypeError(
                f"organization_key {self.organization_key} is type {type(self.organization_key)}, but must be a str"
            )
        if not isinstance(self.model_id, str):
            raise TypeError(
                f"model_id {self.model_id} is type {type(self.model_id)}, but must be a str"
            )
        if not isinstance(self.prediction_ids, (pd.DataFrame, pd.Series)):
            raise TypeError(
                f"prediction_ids is type {type(self.prediction_ids)}, but expect one of: pd.DataFrame, pd.Series"
            )
        if not isinstance(self.feature_importances, (pd.DataFrame, pd.Series)):
            raise TypeError(
                f"feature_importances is type: {type(self.feature_importances)}, but expects one of: pd.DataFrame, "
                f"pd.Series "
            )

        if self.feature_importances is not None:
            self._validate_feature_importances()

        if self.feature_importances.shape[0] != self.prediction_ids.shape[0]:
            raise ValueError(
                f"feature_importances contains {self.feature_importances.shape[0]} elements, but must have the same "
                f"as predictions_ids: {self.feature_importances.shape[0]}. "
            )

    def _validate_feature_importances(self):
        if self.feature_importances is None:
            return
        if not isinstance(self.feature_importances, pd.DataFrame):
            raise TypeError(
                f"feature_importances is type {type(self.feature_importances)}, but expect type pd.DataFrame."
            )
        if self.feature_importances.shape[0] != self.prediction_ids.shape[0]:
            raise ValueError(
                f"feature_importances has {self.feature_importances.shape[0]} sets of values, but must match size of "
                f"predictions_ids: {self.prediction_ids.shape[0]}. "
            )
        if isinstance(
            self.feature_importances.columns, pd.core.indexes.numeric.NumericIndex
        ):
            raise TypeError(
                f"features.columns is of type {type(self.feature_importances.columns)}, but expect elements to be str."
            )
        for name in self.feature_importances.columns:
            if not isinstance(name, str):
                raise TypeError(
                    f"features_importances.column {name} is type {type(name)}, but expect str"
                )

    def build_proto(self):
        columns = self.feature_importances.columns
        self.prediction_ids = self.prediction_ids.to_numpy()
        self.feature_importances = self.feature_importances.to_numpy()
        records = []
        for i, v in enumerate(self.prediction_ids):
            pred_id = v if isinstance(v, str) else v[0]
            if not isinstance(pred_id, (str, bytes)):
                raise TypeError(
                    f"prediction_id {pred_id} is type {type(pred_id)}, but expected one of: str, bytes"
                )

            converted_fi = {
                name: self.feature_importances[i][column]
                for column, name in enumerate(columns)
            }
            fi = public__pb2.FeatureImportances(feature_importances=converted_fi)

            records.append(
                public__pb2.Record(prediction_id=pred_id, feature_importances=fi)
            )
        return get_bulk_records(
            organization_key=self.organization_key,
            model_id=self.model_id,
            model_version=None,
            records=bundle_records(records),
        )


class PreProductionRecords(BaseRecord, ABC):
    def __init__(
        self,
        organization_key: str,
        model_id: str,
        model_version: str,
        prediction_labels: Union[pd.DataFrame, pd.Series],
        actual_labels: Union[pd.DataFrame, pd.Series],
        features: Optional[pd.DataFrame] = None,
        model_type: Optional[ModelTypes] = None,
        prediction_scores: Optional[Union[pd.DataFrame, pd.Series]] = None,
    ):
        if model_type is None:
            if prediction_scores is None:
                self.model_type = infer_model_type(prediction_labels[0])
            else:
                self.model_type = ModelTypes.SCORE_CATEGORICAL
        super().__init__(
            organization_key=organization_key, model_id=model_id, model_type=model_type
        )
        self.model_version = model_version
        self.features = features
        self.prediction_labels = prediction_labels
        self.prediction_scores = prediction_scores
        self.actual_labels = actual_labels

    def _validate_preprod_inputs(self):
        self._base_validation()
        if not isinstance(self.model_version, str):
            raise TypeError(
                f"model_version {self.model_version} is type {type(self.model_version)}, but must be a str"
            )
        if not isinstance(self.prediction_labels, (pd.DataFrame, pd.Series)):
            raise TypeError(
                f"prediction_labels is type {type(self.prediction_labels)}, but expects one of: pd.DataFrame, pd.Series"
            )
        if self.prediction_labels.shape[0] != self.actual_labels.shape[0]:
            raise ValueError(
                f"prediction_labels contains {self.prediction_labels.shape[0]} elements, but must have the same as "
                f"actual_labels: {self.actual_labels.shape[0]}. "
            )
        if self.model_type == ModelTypes.SCORE_CATEGORICAL:
            if not isinstance(self.prediction_scores, (pd.DataFrame, pd.Series)):
                raise TypeError(
                    f"prediction_scores is type {type(self.prediction_scores)}, but expects one of: pd.DataFrame, pd.Series"
                )
            if self.prediction_scores.shape[0] != self.prediction_labels.shape[0]:
                raise ValueError(
                    f"prediction_scores contains {self.prediction_scores.shape[0]} elements, but must have the same as "
                    f"prediction_labels: {self.prediction_scores.shape[0]}."
                )
        if self.features is None:
            return
        if not isinstance(self.features, pd.DataFrame):
            raise TypeError(
                f"features is type {type(self.features)}, but expect type pd.DataFrame."
            )
        if self.features.shape[0] != self.prediction_labels.shape[0]:
            raise ValueError(
                f"features has {self.features.shape[0]} sets of features, but must match size of prediction_labels: "
                f"{self.prediction_labels.shape[0]}. "
            )
        for name in self.features.columns:
            if not isinstance(name, str):
                raise TypeError(
                    f"features.column {name} is type {type(name)}, but expect str"
                )

    def _normalize_inputs(self):
        """Converts inputs from DataFrames, Series, lists to numpy arrays or lists for consistent iterations
        downstream."""
        if isinstance(self.prediction_labels, (pd.DataFrame, pd.Series)):
            self.prediction_labels = self.prediction_labels.to_numpy()
        if isinstance(self.actual_labels, (pd.DataFrame, pd.Series)):
            self.actual_labels = self.actual_labels.to_numpy()
        if isinstance(self.prediction_scores, (pd.DataFrame, pd.Series)):
            self.prediction_scores = self.prediction_scores.to_numpy()
        if isinstance(self.features, pd.DataFrame):
            self.feature_names = self.features.columns
            self.features = self.features.to_numpy()


class TrainingRecords(PreProductionRecords):
    def __init__(
        self,
        organization_key: str,
        model_id: str,
        model_version: str,
        prediction_labels: Union[pd.DataFrame, pd.Series],
        actual_labels: Union[pd.DataFrame, pd.Series],
        features: Optional[pd.DataFrame] = None,
        model_type: Optional[ModelTypes] = None,
        prediction_scores: Optional[Union[pd.DataFrame, pd.Series]] = None,
    ):
        super().__init__(
            organization_key=organization_key,
            model_id=model_id,
            model_type=model_type,
            model_version=model_version,
            features=features,
            prediction_labels=prediction_labels,
            prediction_scores=prediction_scores,
            actual_labels=actual_labels,
        )

    def validate_inputs(self):
        self._validate_preprod_inputs()

    def build_proto(self):
        records = []
        self._normalize_inputs()
        for row, v in enumerate(self.prediction_labels):
            a = public__pb2.Actual(
                label=self._get_label(value=self.actual_labels[row], name="actual")
            )
            score = (
                None
                if self.model_type != ModelTypes.SCORE_CATEGORICAL
                else self.prediction_scores[row]
            )
            p = public__pb2.Prediction(
                label=self._get_label(value=v, name="prediction", score=score),
                model_version=self.model_version,
            )

            if self.features is not None:
                converted_feats = {}
                for column, name in enumerate(self.feature_names):
                    val = get_value_object(value=self.features[row][column], name=name)
                    if val is not None:
                        converted_feats[name] = val
                feats = public__pb2.Prediction(features=converted_feats)
                p.MergeFrom(feats)

            panda = public__pb2.PredictionAndActual(
                prediction=p,
                actual=a,
            )
            r = public__pb2.Record(
                organization_key=self.organization_key,
                model_id=self.model_id,
                prediction_and_actual=panda,
            )

            t = public__pb2.PreProductionRecord(
                training_record=public__pb2.PreProductionRecord.TrainingRecord(record=r)
            )
            records.append(t)
        return bundle_records(records)


class ValidationRecords(PreProductionRecords):
    def __init__(
        self,
        organization_key: str,
        model_id: str,
        model_version: str,
        batch_id: str,
        prediction_labels: Union[pd.DataFrame, pd.Series],
        actual_labels: Union[pd.DataFrame, pd.Series],
        features: Optional[pd.DataFrame] = None,
        model_type: Optional[ModelTypes] = None,
        prediction_scores: Optional[Union[pd.DataFrame, pd.Series]] = None,
    ):
        super().__init__(
            organization_key=organization_key,
            model_id=model_id,
            model_type=model_type,
            model_version=model_version,
            features=features,
            prediction_labels=prediction_labels,
            prediction_scores=prediction_scores,
            actual_labels=actual_labels,
        )
        self.batch_id = batch_id

    def validate_inputs(self):
        self._validate_preprod_inputs()
        if not isinstance(self.batch_id, str):
            raise TypeError(
                f"batch_id {self.batch_id} is type {type(self.batch_id)}, but must be a str"
            )

    def build_proto(self):
        records = []
        self._normalize_inputs()
        for row, v in enumerate(self.prediction_labels):
            a = public__pb2.Actual(
                label=self._get_label(value=self.actual_labels[row], name="actual")
            )
            score = (
                None
                if self.model_type != ModelTypes.SCORE_CATEGORICAL
                else self.prediction_scores[row]
            )
            p = public__pb2.Prediction(
                label=self._get_label(value=v, name="prediction", score=score),
                model_version=self.model_version,
            )

            if self.features is not None:
                converted_feats = {}
                for column, name in enumerate(self.feature_names):
                    val = get_value_object(value=self.features[row][column], name=name)
                    if val is not None:
                        converted_feats[name] = val
                feats = public__pb2.Prediction(features=converted_feats)
                p.MergeFrom(feats)

            panda = public__pb2.PredictionAndActual(
                prediction=p,
                actual=a,
            )
            r = public__pb2.Record(
                organization_key=self.organization_key,
                model_id=self.model_id,
                prediction_and_actual=panda,
            )
            v = public__pb2.PreProductionRecord(
                validation_record=public__pb2.PreProductionRecord.ValidationRecord(
                    batch_id=self.batch_id, record=r
                ),
            )
            records.append(v)
        return bundle_records(records)
