import base64
import logging
import sys
import tempfile
from typing import Optional

import pandas as pd
import pandas.api.types as ptypes
import pyarrow as pa
import requests

import arize.pandas.validation.errors as err
from arize import public_pb2 as pb
from arize.__init__ import __version__
from arize.pandas.validation.validator import Validator
from arize.utils.types import ModelTypes, Environments
from arize.utils.types import Schema
from arize.utils.utils import reconstruct_url

logger = logging.getLogger(__name__)
if hasattr(sys, "ps1"):
    # for python interactive mode
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setLevel(logging.INFO)
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)


class Client:
    """
    Arize API Client to  model predictions and actuals to the Arize AI platform from pandas.DataFrames
    """

    def __init__(
        self,
        api_key: str,
        space_key: str,
        uri: Optional[str] = "https://api.arize.com/v1",
    ) -> None:
        """
        :param api_key (str): api key associated with your account with Arize AI.
        :param space_key (str): space key in Arize AI.
        :param uri (str, optional): uri to send your records to Arize AI. Defaults to "https://api.arize.com/v1".
        """
        self._api_key = api_key
        self._space_key = space_key
        self._files_uri = uri + "/pandas_arrow"

    def log(
        self,
        dataframe: pd.DataFrame,
        schema: Schema,
        environment: Environments,
        model_id: str,
        model_type: ModelTypes,
        model_version: Optional[str] = None,
        batch_id: Optional[str] = None,
        sync: Optional[bool] = False,
        validate: Optional[bool] = True,
        path: Optional[str] = None,
        surrogate_explainability: Optional[bool] = False,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = False,
    ) -> requests.Response:
        """Logs a pandas dataframe containing inferences to Arize via a POST request. Returns a :class:`Response` object from the Requests HTTP library.
        :param dataframe (pd.DataFrame): The dataframe containing inferences.
        :param schema (Schema): A Schema instance that maps model inference data fields to column names in the provided dataframe.
        :param environment (Environments): The environment that this dataframe is for (Production, Training, Validation).
        :param model_id (str): The unique identifier for your model.
        :param model_type (ModelTypes): Declared what model type this prediction is for.
        :param model_version (str, optional): Used to group together a subset of predictions and actuals for a given model_id. Defaults to None.
        :param batch_id (str, optional): Used to distinguish different batch of data under the same model_id and  model_version. Defaults to None.
        :param sync (bool, optional): When sync is set to True, the log call will block, or wait, until the data has been successfully ingested by the platform and immediately return the status of the log. Defaults to False.
        :param validate (bool, optional): When set to True, validation is run before sending data. Defaults to True.
        :param path (str, optional): Temporary directory/file to store the serialized data in binary before sending to Arize.
        :param surrogate_explainability (bool, optional): Computes feature importance values using the surrogate explainability method. This requires that the arize module is installed with the [MimicExplainer] option. If feature importance values are already specified by the shap_values_column_names attribute in the Schema, this module will not run. Defaults to False.
        :param timeout (float, optional): You can stop waiting for a response after a given number of seconds with the timeout parameter. Defaults to None.
        """

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
            if verbose:
                print("Performing parameters validation.")
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

        if verbose:
            print("Removing unnecessary columns.")
        dataframe = self._remove_extraneous_columns(dataframe, schema)

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

        if surrogate_explainability:
            if verbose:
                print("Running surrogate_explainability.")
            try:
                from arize.pandas.surrogate_explainer.mimic import Mimic
            except ImportError:
                raise ImportError(
                    "To enable surrogate explainability, "
                    "the arize module must be installed with the MimicExplainer option: pip install 'arize[MimicExplainer]'."
                )
            if schema.shap_values_column_names:
                logger.info(
                    "surrogate_explainability=True has no effect "
                    "because shap_values_column_names is already specified in schema."
                )
            elif schema.feature_column_names is None or (
                hasattr(schema.feature_column_names, "__len__")
                and len(schema.feature_column_names) == 0
            ):
                logger.info(
                    "surrogate_explainability=True has no effect "
                    "because feature_column_names is empty or not specified in schema."
                )
            else:
                dataframe, schema = Mimic.augment(
                    df=dataframe, schema=schema, model_type=model_type
                )

        # pyarrow will err if a mixed type column exist in the dataset even if
        # the column is not specified in schema. Caveat: There may be other
        # error conditions that we're currently not aware of.
        try:
            if verbose:
                print("Getting pyarrow schema from pandas dataframe.")
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
            if verbose:
                print("Performing types validation.")
            errors = Validator.validate_types(
                model_type=model_type,
                schema=schema,
                pyarrow_schema=pa_schema,
            )
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)
            if verbose:
                print("Performing values validation.")
            errors = Validator.validate_values(
                dataframe=dataframe, schema=schema, model_type=model_type
            )
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

        if verbose:
            print("Getting pyarrow table from pandas dataframe.")
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

        if model_type == ModelTypes.NUMERIC:
            s.constants.model_type = pb.Schema.ModelType.NUMERIC
        elif model_type == ModelTypes.SCORE_CATEGORICAL:
            s.constants.model_type = pb.Schema.ModelType.SCORE_CATEGORICAL
        elif model_type == ModelTypes.RANKING:
            s.constants.model_type = pb.Schema.ModelType.RANKING

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

        if schema.embedding_feature_column_names is not None:
            for embedding_feature_column_names in schema.embedding_feature_column_names:
                embedding_name = (
                    embedding_feature_column_names.vector_column_name
                )  # how it will show in the UI
                s.arrow_schema.embedding_feature_column_names_map[
                    embedding_name
                ].vector_column_name = embedding_feature_column_names.vector_column_name
                if embedding_feature_column_names.data_column_name:
                    s.arrow_schema.embedding_feature_column_names_map[
                        embedding_name
                    ].data_column_name = embedding_feature_column_names.data_column_name
                if embedding_feature_column_names.link_to_data_column_name:
                    s.arrow_schema.embedding_feature_column_names_map[
                        embedding_name
                    ].link_to_data_column_name = (
                        embedding_feature_column_names.link_to_data_column_name
                    )

        if schema.tag_column_names is not None:
            s.arrow_schema.tag_column_names.extend(schema.tag_column_names)

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

        if schema.prediction_group_id_column_name is not None:
            s.arrow_schema.prediction_group_id_column_name = (
                schema.prediction_group_id_column_name
            )

        if schema.rank_column_name is not None:
            s.arrow_schema.rank_column_name = schema.rank_column_name

        if verbose:
            print("Serializing schema.")
        base64_schema = base64.b64encode(s.SerializeToString())

        # limit the potential size of http headers to under 64 kilobytes
        if len(base64_schema) > 63000:
            raise ValueError(
                "The schema (after removing unnecessary columns) is too large."
            )

        if path is None:
            f = tempfile.NamedTemporaryFile()
            tmp_file = f.name
        else:
            tmp_file = path

        try:
            if verbose:
                print("Writing table to temporary file: ", tmp_file)
            writer = pa.ipc.new_stream(tmp_file, pa_schema)
            writer.write_table(ta, max_chunksize=65536)
            writer.close()
            if verbose:
                print("Sending file to Arize")
            response = self._post_file(tmp_file, base64_schema, sync, timeout)
        finally:
            if path is None:
                f.close()

        try:
            logger.info(f"Success! Check out your data at {reconstruct_url(response)}")
        except:
            pass

        return response

    def _post_file(
        self,
        path: str,
        schema: bytes,
        sync: Optional[bool],
        timeout: Optional[float] = None,
    ) -> requests.Response:
        with open(path, "rb") as f:
            headers = {
                "authorization": self._api_key,
                "space": self._space_key,
                "schema": schema,
                "sdk-version": __version__,
                "sdk": "py",
            }
            if sync:
                headers["sync"] = "1"
            return requests.post(
                self._files_uri,
                timeout=timeout,
                data=f,
                headers=headers,
            )

    def _remove_extraneous_columns(
        self, df: pd.DataFrame, schema: Schema
    ) -> pd.DataFrame:
        cols_to_keep = set()

        for field in schema.__dataclass_fields__:
            if field.endswith("column_name"):
                col = getattr(schema, field)
                if col is not None:
                    cols_to_keep.add(col)

        if schema.feature_column_names is not None:
            for col in schema.feature_column_names:
                cols_to_keep.add(col)

        if schema.embedding_feature_column_names is not None:
            for emb_col_names in schema.embedding_feature_column_names:
                cols_to_keep.add(emb_col_names.vector_column_name)
                if emb_col_names.data_column_name is not None:
                    cols_to_keep.add(emb_col_names.data_column_name)
                if emb_col_names.link_to_data_column_name is not None:
                    cols_to_keep.add(emb_col_names.link_to_data_column_name)

        if schema.tag_column_names is not None:
            for col in schema.tag_column_names:
                cols_to_keep.add(col)

        if schema.shap_values_column_names is not None:
            for col in schema.shap_values_column_names.values():
                cols_to_keep.add(col)

        return df[df.columns.intersection(cols_to_keep)]
