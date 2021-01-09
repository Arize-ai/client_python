import math

from abc import ABC, abstractmethod
from google.protobuf.timestamp_pb2 import Timestamp
import pandas as pd

from arize import public_pb2 as public__pb2


class BaseRecord(ABC):
    def __init__(self, organization_key, model_id):
        self.organization_key = organization_key
        self.model_id = model_id

    @abstractmethod
    def validate_inputs(self):
        pass

    @abstractmethod
    # combine _build_proto with validate_inputs so that build proto isn't called when the
    # inputs haven't been validated
    def _build_proto(self):
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

    def _get_timestamp(self, time_overwrite=None):
        ts = None
        if time_overwrite is not None:
            time = self._convert_element(time_overwrite)
            if not isinstance(time_overwrite, int):
                raise TypeError(
                    f"time_overwrite {time_overwrite} is type {type(time_overwrite)}, but expects int. (Unix epoch time in seconds)"
                )
            ts = Timestamp()
            ts.FromSeconds(time)
        return ts

    @staticmethod
    def _convert_element(value):
        """ converts scalar or array to python native """
        val = getattr(value, "tolist", lambda: value)()
        # Check if it's a list since elements from pd indices are converted to a scalar
        # whereas pd series/dataframe elements are converted to list of 1 with the native value
        if isinstance(val, list):
            return val[0]
        return val

    def _get_value(self, name: str, value):
        if isinstance(value, public__pb2.Value):
            return value
        val = self._convert_element(value)
        if isinstance(val, (str, bool)):
            return public__pb2.Value(string=str(val))
        if isinstance(val, int):
            return public__pb2.Value(int=val)
        if isinstance(val, float):
            return public__pb2.Value(double=val)
        else:
            raise TypeError(
                f'feature "{name}" = {value} is type {type(value)}, but must be one of: bool, str, float, int.'
            )

    def _get_label(self, name: str, value):
        if isinstance(value, public__pb2.Label):
            return value
        val = self._convert_element(value)
        if isinstance(val, bool):
            return public__pb2.Label(binary=val)
        if isinstance(val, str):
            return public__pb2.Label(categorical=val)
        if isinstance(val, (int, float)):
            return public__pb2.Label(numeric=val)
        else:
            raise TypeError(
                f"{name}_label = {value} of type {type(value)}. Must be one of bool, str, float/int"
            )


class Prediction(BaseRecord):
    def __init__(
        self,
        organization_key,
        model_id,
        model_version,
        prediction_id,
        prediction_label,
        features,
        time_overwrite,
    ):
        super().__init__(organization_key=organization_key, model_id=model_id)
        self.model_version = model_version
        self.prediction_id = prediction_id
        self.prediction_label = prediction_label
        self.features = features
        self.time_overwrite = time_overwrite

    def validate_inputs(self):
        self._base_validation()
        if not isinstance(
            self._convert_element(self.prediction_label), (str, bool, float, int)
        ):
            raise TypeError(
                f"prediction_label {self.prediction_label} has type {type(self.prediction_label)}, but must be one of: str, bool, float, int"
            )
        if self.features is not None and bool(self.features):
            for k, v in self.features.items():
                if not isinstance(self._convert_element(v), (str, bool, float, int)):
                    raise TypeError(
                        f"feature {k} with value {v} is type {type(v)}, but expected one of: str, bool, float, int"
                    )
        if self.time_overwrite is not None and not isinstance(self.time_overwrite, int):
            raise TypeError(
                f"time_overwrite {self.time_overwrite} is type {type(self.time_overwrite)} but expected int"
            )

    def _build_proto(self):
        p = public__pb2.Prediction(
            label=self._get_label(value=self.prediction_label, name="prediction")
        )
        if self.features is not None:
            feats = public__pb2.Prediction(
                features={
                    k: self._get_value(value=v, name=k)
                    for (k, v) in self.features.items()
                }
            )
            p.MergeFrom(feats)
        if self.model_version is not None:
            p.model_version = self.model_version
        if self.time_overwrite is not None:
            p.timestamp.MergeFrom(self._get_timestamp(self.time_overwrite))
        return public__pb2.Record(
            organization_key=self.organization_key,
            model_id=self.model_id,
            prediction_id=self.prediction_id,
            prediction=p,
        )


class Actual(BaseRecord):
    def __init__(self, organization_key, model_id, prediction_id, actual_label):
        super().__init__(organization_key=organization_key, model_id=model_id)
        self.prediction_id = prediction_id
        self.actual_label = actual_label

    def validate_inputs(self):
        self._base_validation()
        if not isinstance(
            self._convert_element(self.actual_label), (str, bool, float, int)
        ):
            raise TypeError(
                f"actual_label {self.actual_label} has type {type(self._convert_element(self.actual_label))}, but must be one of: str, bool, float, int"
            )

    def _build_proto(self):
        a = public__pb2.Actual(
            label=self._get_label(value=self.actual_label, name="actual")
        )
        return public__pb2.Record(
            organization_key=self.organization_key,
            model_id=self.model_id,
            prediction_id=self.prediction_id,
            actual=a,
        )

class FeatureImportances(BaseRecord):
    def __init__(
        self,
        organization_key,
        model_id,
        prediction_id,
        feature_importances,
    ):
        super().__init__(organization_key=organization_key, model_id=model_id)
        self.prediction_id = prediction_id
        self.feature_importances = feature_importances

    def validate_inputs(self):
        self._base_validation()
        if self.feature_importances is not None and bool(self.feature_importances):
            for k, v in self.feature_importances.items():
                if not isinstance(v, (float)):
                    raise TypeError(
                        f"feature {k} with value {v} is type {type(v)}, but expected one of: float"
                    )
        elif self.feature_importances is None or len(self.feature_importances) == 0:
            raise ValueError(
                f"at least one feature importance value must be provided"
            )

    def _build_proto(self):
        fi = public__pb2.FeatureImportances()

        if self.feature_importances is not None:
            featImportances = public__pb2.FeatureImportances(
                feature_importances=self.feature_importances
            )
            fi.MergeFrom(featImportances)
        return public__pb2.Record(
            organization_key=self.organization_key,
            model_id=self.model_id,
            prediction_id=self.prediction_id,
            feature_importances=fi,
        )

class BaseBulkRecord(BaseRecord, ABC):
    MAX_BYTES_PER_BULK_RECORD = 100000

    prediction_labels, actual_labels, time_overwrite, features = None, None, None, None

    def __init__(self, organization_key, model_id, prediction_ids):
        super().__init__(organization_key, model_id)
        self.organization_key = organization_key
        self.model_id = model_id
        self.prediction_ids = prediction_ids

    def _base_bulk_validation(self):
        self._base_validation()
        if not isinstance(self.prediction_ids, (pd.DataFrame, pd.Series)):
            raise TypeError(
                f"prediction_ids is type {type(self.prediction_ids)}, but expect one of: pd.DataFrame, pd.Series"
            )

    def _bundle_records(self, records, model_version):
        recs_per_msg = self._num_chunks(records)
        recs = {
            (i, i + recs_per_msg): records[i : i + recs_per_msg]
            for i in range(0, len(records), recs_per_msg)
        }
        for k, r in recs.items():
            recs[k] = public__pb2.BulkRecord(
                records=r,
                organization_key=self.organization_key,
                model_id=self.model_id,
                model_version=model_version,
            )
        return recs

    def _num_chunks(self, records):
        total_bytes = sum(r.ByteSize() for r in records)
        num_of_bulk = math.ceil(total_bytes / self.MAX_BYTES_PER_BULK_RECORD)
        return math.ceil(len(records) / num_of_bulk)

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


class BulkPrediction(BaseBulkRecord):
    def __init__(
        self,
        organization_key,
        model_id,
        model_version,
        prediction_ids,
        prediction_labels,
        features,
        feature_names_overwrite,
        time_overwrite,
    ):
        super().__init__(organization_key, model_id, prediction_ids)
        self.model_version = model_version
        self.prediction_labels = prediction_labels
        self.features = features
        self.feature_names_overwrite = feature_names_overwrite
        self.time_overwrite = time_overwrite

    def validate_inputs(self):
        self._base_bulk_validation()
        if not isinstance(self.prediction_labels, (pd.DataFrame, pd.Series)):
            raise TypeError(
                f"prediction_labels is type: {type(self.prediction_labels)}, but expects one of: pd.DataFrame, pd.Series"
            )
        if self.prediction_labels.shape[0] != self.prediction_ids.shape[0]:
            raise ValueError(
                f"prediction_labels contains {self.prediction_labels.shape[0]} elements, but must have the same as predictions_ids: {self.prediction_ids.shape[0]}."
            )
        self._validate_features()
        self._validate_time_overwrite()

    def _build_proto(self):
        self._normalize_inputs()
        records = []
        for row, v in enumerate(self.prediction_ids):
            pred_id = v if isinstance(v, str) else v[0]
            if not isinstance(pred_id, (str, bytes)):
                raise TypeError(
                    f"prediction_id {pred_id} is type {type(pred_id)}, but expected one of: str, bytes"
                )
            p = public__pb2.Prediction(
                label=self._get_label(
                    value=self.prediction_labels[row], name="prediction"
                )
            )
            if self.features is not None:
                converted_feats = {
                    name: self._get_value(value=self.features[row][column], name=name)
                    for column, name in enumerate(self.feature_names)
                }
                feats = public__pb2.Prediction(features=converted_feats)
                p.MergeFrom(feats)
            if self.time_overwrite is not None:
                p.timestamp.MergeFrom(self._get_timestamp(self.time_overwrite[row]))

            records.append(public__pb2.Record(prediction_id=pred_id, prediction=p))
        return self._bundle_records(records, self.model_version)

    def _validate_features(self):
        if self.features is None:
            return
        if not isinstance(self.features, pd.DataFrame):
            raise TypeError(
                f"features is type {type(self.features)}, but expect type pd.DataFrame."
            )
        if self.features.shape[0] != self.prediction_ids.shape[0]:
            raise ValueError(
                f"features has {self.features.shape[0]} sets of features, but must match size of predictions_ids: {self.prediction_ids.shape[0]}."
            )
        if self.feature_names_overwrite is not None:
            if len(self.features.columns) != len(self.feature_names_overwrite):
                raise ValueError(
                    f"feature_names_overwrite has len:{len(self.feature_names_overwrite)}, but expects the same number of columns in features dataframe: {len(self.features.columns)}."
                )
        else:
            if isinstance(self.features.columns, pd.core.indexes.numeric.NumericIndex):
                raise TypeError(
                    f"features.columns is of type {type(self.features.columns)}, but expect elements to be str. Alternatively, feature_names_overwrite must be present."
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
                    f"time_overwrite has {self.time_overwrite.shape[0]} elements, but must have same number of elements as prediction_ids: {expected_count}."
                )
        elif isinstance(self.time_overwrite, list):
            if len(self.time_overwrite) != expected_count:
                raise ValueError(
                    f"time_overwrite has length {len(self.time_overwrite)} but must have same number of elements as prediction_ids: {expected_count}."
                )
        else:
            raise TypeError(
                f"time_overwrite is type {type(self.time_overwrite)}, but expected one of: pd.Series, list<int>"
            )


class BulkActual(BaseBulkRecord):
    def __init__(self, organization_key, model_id, prediction_ids, actual_labels):
        super().__init__(organization_key, model_id, prediction_ids)
        self.actual_labels = actual_labels

    def validate_inputs(self):
        self._base_bulk_validation()
        if not isinstance(self.actual_labels, (pd.DataFrame, pd.Series)):
            raise TypeError(
                f"actual_labels is type: {type(self.actual_labels)}, but expects one of: pd.DataFrame, pd.Series"
            )
        if self.actual_labels.shape[0] != self.prediction_ids.shape[0]:
            raise ValueError(
                f"actual_labels contains {self.actual_labels.shape[0]} elements, but must have the same as predictions_ids: {self.prediction_ids.shape[0]}."
            )

    def _build_proto(self):
        self._normalize_inputs()
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
        return self._bundle_records(records, None)


class BulkFeatureImportances(BaseBulkRecord):
    def __init__(self, organization_key, model_id, prediction_ids, feature_importances):
        super().__init__(organization_key, model_id, prediction_ids)
        self.feature_importances = feature_importances

    def validate_inputs(self):
        self._base_bulk_validation()
        if not isinstance(self.feature_importances, (pd.DataFrame, pd.Series)):
            raise TypeError(
                f"feature_importances is type: {type(self.feature_importances)}, but expects one of: pd.DataFrame, pd.Series"
            )
        if self.feature_importances is not None and bool(self.feature_importances):
            for k, v in self.feature_importances.items():
                if not isinstance(v, (float)):
                    raise TypeError(
                        f"feature {k} with value {v} is type {type(v)}, but expected one of: float"
                    )
        if self.feature_importances.shape[0] != self.prediction_ids.shape[0]:
            raise ValueError(
                f"feature_importances contains {self.feature_importances.shape[0]} elements, but must have the same as predictions_ids: {self.feature_importances.shape[0]}."
            )

    def _build_proto(self):
        self._normalize_inputs()
        self.fi = self.feature_importances.to_numpy()
        records = []
        for i, v in enumerate(self.prediction_ids):
            pred_id = v if isinstance(v, str) else v[0]
            if not isinstance(pred_id, (str, bytes)):
                raise TypeError(
                    f"prediction_id {pred_id} is type {type(pred_id)}, but expected one of: str, bytes"
                )

            converted_fi = {
                name: self.fi[i][column]
                for column, name in enumerate(self.feature_importances.columns)
            }
            fi = public__pb2.FeatureImportances(feature_importances=converted_fi)

            records.append(public__pb2.Record(prediction_id=pred_id, feature_importances=fi))
        return self._bundle_records(records, None)
