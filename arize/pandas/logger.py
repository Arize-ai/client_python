import base64
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd
import pandas.api.types as ptypes

import pyarrow as pa
import requests
import tempfile

from arize import public_pb2 as pb
from arize.__init__ import __version__
from arize.utils.types import ModelTypes, Environments
import arize.pandas.validation.errors as err
from arize.pandas.validation.validator import Validator


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
    actual_numeric_sequence_column_name: Optional[str] = None


class Client:
    def __init__(
        self, api_key: str, organization_key: str, uri: str = "https://api.arize.com/v1"
    ):
        self._api_key = api_key
        self._organization_key = organization_key
        self._files_uri = uri + "/pandas_arrow"

    def log(
        self,
        dataframe: pd.DataFrame,
        model_id: str,
        model_type: ModelTypes,
        environment: Environments,
        schema: Schema,
        model_version: Optional[str] = None,
        batch_id: Optional[str] = None,
        sync: Optional[bool] = False,
        validate: Optional[bool] = True,
        path: Optional[str] = None,
    ) -> requests.Response:
        logger = logging.getLogger(__name__)

        if model_id is not None and not isinstance(model_id, str):
            try:
                model_id = str(model_id)
            except:
                logger.error("model_id must be convertible to a string.")
                raise

        if model_version is not None and not isinstance(model_version, str):
            try:
                model_version = str(model_version)
            except:
                logger.error("model_version must be convertible to a string.")
                raise

        if batch_id is not None and not isinstance(batch_id, str):
            try:
                batch_id = str(batch_id)
            except:
                logger.error("batch_id must be convertible to a string.")
                raise

        if validate:
            errors = Validator.validate_params(
                dataframe=dataframe,
                model_id=model_id,
                model_type=model_type,
                environment=environment,
                schema=schema,
                model_version=model_version,
                batch_id=batch_id,
            )
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

        # always validate pd.Category is not present, if yes, convert to string
        has_cat_col = any([ptypes.is_categorical_dtype(x) for x in dataframe.dtypes])
        if has_cat_col:
            cat_cols = [
                col_name
                for col_name, col_cat in dataframe.dtypes.items()
                if col_cat.name == "category"
            ]
            cat_str_map = dict(zip(cat_cols, ["str"] * len(cat_cols)))
            dataframe = dataframe.astype(cat_str_map)

        # pyarrow will err if a mixed type column exist in the dataset even if
        # the column is not specified in schema. Caveat: There may be other
        # error conditions that we're currently not aware of.
        try:
            pa_schema = pa.Schema.from_pandas(dataframe)
        except pa.ArrowInvalid as e:
            logger.error(
                "The dataframe needs to convert to pyarrow but has failed to do so. "
                "There may be unrecognized data types in the dataframe. "
                "Another reason may be that a column in the dataframe has a mix of strings and numbers, "
                "in which case you may want to convert the strings in that column to NaN. "
                "See https://docs.arize.com/arize/api-reference/python-sdk/arize.pandas/mixed-types"
            )
            raise

        if validate:
            errors = Validator.validate_types(
                model_type=model_type,
                schema=schema,
                pyarrow_schema=pa_schema,
            )
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

            errors = Validator.validate_values(
                dataframe=dataframe,
                schema=schema,
            )
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

        ta = pa.Table.from_pandas(dataframe)

        s = pb.Schema()
        s.constants.model_id = model_id

        if model_version is not None:
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
            s.arrow_schema.prediction_label_column_name = (
                schema.prediction_label_column_name
            )

        if schema.prediction_score_column_name is not None:
            s.arrow_schema.prediction_score_column_name = (
                schema.prediction_score_column_name
            )

        if schema.feature_column_names is not None:
            s.arrow_schema.feature_column_names.extend(schema.feature_column_names)

        if schema.actual_label_column_name is not None:
            s.arrow_schema.actual_label_column_name = schema.actual_label_column_name

        if schema.actual_score_column_name is not None:
            s.arrow_schema.actual_score_column_name = schema.actual_score_column_name

        if schema.shap_values_column_names is not None:
            s.arrow_schema.shap_values_column_names.update(
                schema.shap_values_column_names
            )

        if schema.actual_numeric_sequence_column_name is not None:
            s.arrow_schema.actual_numeric_sequence_column_name = (
                schema.actual_numeric_sequence_column_name
            )

        base64_schema = base64.b64encode(s.SerializeToString())

        if path is None:
            f = tempfile.NamedTemporaryFile()
            tmp_file = f.name
        else:
            tmp_file = path

        try:
            writer = pa.ipc.new_stream(tmp_file, pa_schema)
            writer.write_table(ta, max_chunksize=65536)
            writer.close()
            response = self._post_file(tmp_file, base64_schema, sync)
        finally:
            if path is None:
                f.close()

        return response

    def _post_file(
        self, path: str, schema: bytes, sync: Optional[bool]
    ) -> requests.Response:
        with open(path, "rb") as f:
            headers = {
                "authorization": self._api_key,
                "organization": self._organization_key,
                "schema": schema,
                "sdk-version": __version__,
                "sdk": "py",
            }
            if sync:
                headers["sync"] = "1"
            return requests.post(
                self._files_uri,
                data=f,
                headers=headers,
            )
