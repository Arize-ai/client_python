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

# TODO this class hierarchy will go away


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
