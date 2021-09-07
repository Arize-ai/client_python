import time
import requests
from typing import List, Dict
from dataclasses import dataclass
from arize import public_pb2 as pb
from arize.utils.types import ModelTypes, Environments


@dataclass(frozen=True)
class Schema:
    prediction_id_column_name: str
    feature_column_names: List[str]
    timestamp_column_name: str = None
    prediction_label_column_name: str = None
    prediction_score_column_name: str = None
    actual_label_column_name: str = None
    shap_values_column_names: Dict[str, str] = None


class Client:
    def __init__(
            self,
            api_key: str,
            organization_key: str,
            uri="https://api.arize.com/v1"):
        self._api_key = api_key
        self._organization_key = organization_key
        self._files_uri = uri + "/files"

    def log(
        self,
        dataframe,
        path: str,
        model_id: str,
        model_version: str,
        batch_id: str,
        model_type: ModelTypes,
        environment: Environments,
        schema: Schema
    ):
        col_idx = {col: idx for idx, col in enumerate(dataframe.columns)}
        h = pb.FileHeader()

        if model_type is None:
            raise AttributeError("model_type is required")

        if environment is None:
            raise AttributeError("environment is required")

        if environment is Environments.TRAINING:
            if schema.prediction_label_column_name is None or schema.actual_label_column_name is None:
                raise AttributeError("both prediction and actual label must be specified for Training environment")
            if schema.shap_values_column_names is not None:
                raise AttributeError("shap_values are not supported for Training environments")
            h.environment = pb.FileHeader.Environment.TRAINING
        elif environment is Environments.VALIDATION:
            if schema.prediction_label_column_name is None or schema.actual_label_column_name is None:
                raise AttributeError("both prediction and actual label must be specified for Validation environment")
            if schema.shap_values_column_names is not None:
                raise AttributeError("shap_values are not supported for Validation environments")
            if batch_id is None:
                raise AttributeError("batch_id is required for Validation environment")
            h.environment = pb.FileHeader.Environment.VALIDATION
        elif environment is Environments.PRODUCTION:
            h.environment = pb.FileHeader.Environment.PRODUCTION
        else:
            raise AttributeError(f"unknown environment {environment}")

        with open(path, "wb") as f:
            header = h.SerializeToString()
            f.write(len(header).to_bytes(8, "big", signed=False))
            f.write(header)

            current_time = int(time.time())
            for row in dataframe.to_numpy():
                if Environments.TRAINING == environment:
                    msg = pb.PreProductionRecord()
                    r = msg.training_record.record
                elif Environments.VALIDATION == environment:
                    msg = pb.PreProductionRecord()
                    msg.validation_record.batch_id = batch_id
                    r = msg.validation_record.record
                else:
                    msg = pb.Record()
                    r = msg

                r.prediction_id = row[col_idx[schema.prediction_id_column_name]]
                r.model_id = model_id

                t = (
                    int(row[col_idx[schema.timestamp_column_name]])
                    if schema.timestamp_column_name is not None
                    else current_time
                )

                for feature_cn in schema.feature_column_names:
                    row_val = row[col_idx[feature_cn]]
                    feature_val = r.prediction.features[feature_cn]
                    if isinstance(row_val, (str, bool)):
                        feature_val.string = str(row_val)
                    elif isinstance(row_val, int):
                        feature_val.int = row_val
                    elif isinstance(row_val, float):
                        feature_val.double = row_val

                if schema.prediction_label_column_name is not None:
                    r.prediction.timestamp.seconds = t
                    r.prediction.model_version = model_version
                    if model_type is ModelTypes.SCORE_CATEGORICAL:
                        r.prediction.label.score_categorical.categorical = row[
                            col_idx[schema.prediction_label_column_name]
                        ]
                        r.prediction.label.score_categorical.score = row[
                            col_idx[schema.prediction_score_column_name]
                        ]
                    elif model_type is ModelTypes.CATEGORICAL:
                        r.prediction.label.categorical = row[
                            col_idx[schema.prediction_label_column_name]
                        ]
                    elif model_type is model_type.NUMERIC:
                        r.prediction.label.numeric = row[
                            col_idx[schema.prediction_label_column_name]
                        ]
                    elif model_type is model_type.BINARY:
                        r.prediction.label.binary = row[
                            col_idx[schema.prediction_label_column_name]
                        ]

                if schema.actual_label_column_name is not None:
                    if model_type is ModelTypes.CATEGORICAL:
                        r.actual.label.categorical = row[col_idx[schema.actual_label_column_name]]
                    elif model_type is ModelTypes.NUMERIC:
                        r.actual.label.numeric = row[col_idx[schema.actual_label_column_name]]
                    elif model_type is ModelTypes.BINARY:
                        r.actual.label.binary = row[col_idx[schema.actual_label_column_name]]
                    elif model_type is ModelTypes.SCORE_CATEGORICAL:
                        if isinstance(schema.actual_label_column_name, tuple):
                            r.actual.label.score_categorical.categorical = row[col_idx[schema.actual_label_column_name[0]]]
                            r.actual.label.score_categorical.score = row[col_idx[schema.actual_label_column_name[1]]]
                        else:
                            r.actual.label.score_categorical.categorical = row[col_idx[schema.actual_label_column_name]]

                if schema.shap_values_column_names is not None:
                    for feature_name, shap_values_cn in schema.shap_values_column_names.items():
                        row_val = row[col_idx[shap_values_cn]]
                        r.feature_importances.feature_importances[feature_name] = row_val

                msg_bytes = msg.SerializeToString()
                f.write(len(msg_bytes).to_bytes(8, "big", signed=False))
                f.write(msg_bytes)
        return self._post_file(path)

    def _post_file(self, path):
        with open(path, "rb") as f:
            return requests.post(
                self._files_uri,
                data=f,
                headers={
                    "authorization": self._api_key,
                    "organization": self._organization_key,
                },
            )
