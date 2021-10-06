import base64
from dataclasses import dataclass
from typing import List, Dict, Optional

import pyarrow as pa
import requests
from arize import public_pb2 as pb
from arize.utils.types import ModelTypes, Environments


@dataclass(frozen=True)
class Schema:
    prediction_id_column_name: str
    feature_column_names: Optional[List[str]] = None
    timestamp_column_name: Optional[str] = None
    prediction_label_column_name: Optional[str] = None
    prediction_score_column_name: Optional[str] = None
    actual_label_column_name: Optional[str] = None
    actual_score_column_name: Optional[str] = None
    shap_values_column_names: Optional[Dict[str, str]] = None


class Client:
    def __init__(
            self,
            api_key: str,
            organization_key: str,
            uri="https://api.arize.com/v1"):
        self._api_key = api_key
        self._organization_key = organization_key
        self._files_uri = uri + "/pandas_arrow"

    def log(
        self,
        dataframe,
        path: str,
        model_id: str,
        model_version: str,
        model_type: ModelTypes,
        environment: Environments,
        schema: Schema,
        batch_id: Optional[str] = None,
    ):
        s = pa.Schema.from_pandas(dataframe)
        ta = pa.Table.from_pandas(dataframe)
        writer = pa.ipc.new_stream(path, s)
        writer.write_table(ta, max_chunksize=65536)
        writer.close()

        s = pb.Schema()
        s.constants.model_id = model_id
        s.constants.model_version = model_version

        if environment == Environments.PRODUCTION:
            s.constants.environment = pb.Schema.Environment.PRODUCTION
        elif environment == Environments.VALIDATION:
            s.constants.environment = pb.Schema.Environment.VALIDATION
        elif environment == Environments.TRAINING:
            s.constants.environment = pb.Schema.Environment.TRAINING

        if model_type == ModelTypes.BINARY:
            s.constants.model_type = pb.Schema.ModelType.BINARY
        elif model_type == ModelTypes.NUMERIC:
            s.constants.model_type = pb.Schema.ModelType.NUMERIC
        elif model_type == ModelTypes.CATEGORICAL:
            s.constants.model_type = pb.Schema.ModelType.CATEGORICAL
        elif model_type == ModelTypes.SCORE_CATEGORICAL:
            s.constants.model_type = pb.Schema.ModelType.SCORE_CATEGORICAL

        if batch_id is not None:
            s.constants.batch_id = batch_id

        s.arrow_schema.prediction_id_column_name = schema.prediction_id_column_name

        if schema.timestamp_column_name is not None:
            s.arrow_schema.timestamp_column_name = schema.timestamp_column_name

        if schema.prediction_label_column_name is not None:
            s.arrow_schema.prediction_label_column_name = schema.prediction_label_column_name

        if schema.prediction_score_column_name is not None:
            s.arrow_schema.prediction_score_column_name = schema.prediction_score_column_name

        if schema.feature_column_names is not None:
            s.arrow_schema.feature_column_names.extend(schema.feature_column_names)

        if schema.actual_label_column_name is not None:
            s.arrow_schema.actual_label_column_name = schema.actual_label_column_name

        if schema.actual_score_column_name is not None:
            s.arrow_schema.actual_score_column_name = schema.actual_score_column_name

        if schema.shap_values_column_names is not None:
            s.arrow_schema.shap_values_column_names.update(schema.shap_values_column_names)

        base64_schema = base64.b64encode(s.SerializeToString())
        return self._post_file(path, base64_schema)

    def _post_file(self, path, schema):
        with open(path, "rb") as f:
            return requests.post(
                self._files_uri,
                data=f,
                headers={
                    "authorization": self._api_key,
                    "organization": self._organization_key,
                    "schema": schema,
                },
            )
