# type: ignore[pb2]
import base64
import copy
import json
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pandas.api.types as ptypes
import pyarrow as pa
import requests
from packaging.version import parse as parse_version

from .. import public_pb2 as pb2
from ..__init__ import __version__
from ..utils import proto
from ..utils.constants import (
    API_KEY_ENVVAR_NAME,
    GENERATED_LLM_PARAMS_JSON_COL,
    GENERATED_PREDICTION_LABEL_COL,
    LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
    SPACE_ID_ENVVAR_NAME,
    SPACE_KEY_ENVVAR_NAME,
)
from ..utils.errors import AuthError, InvalidCertificateFile, InvalidTypeAuthKey
from ..utils.logging import log_a_list, logger
from ..utils.types import (
    BaseSchema,
    Environments,
    LLMConfigColumnNames,
    Metrics,
    ModelTypes,
    Schema,
)
from ..utils.utils import (
    get_python_version,
    is_python_version_below_required_min,
    reconstruct_url,
    reset_dataframe_index,
)
from .etl.casting import ETL_ERROR_MESSAGE, ETL_MINIMUM_PANDAS_VERSION, cast_typed_columns
from .tracing.constants import DEFAULT_DATETIME_FMT
from .validation import errors as err
from .validation.validator import Validator

SURROGATE_EXPLAINER_MIN_PYTHON_VERSION = "3.8.0"


INVALID_ARROW_CONVERSION_MSG = (
    "The dataframe needs to convert to pyarrow but has failed to do so. "
    "There may be unrecognized data types in the dataframe. "
    "Another reason may be that a column in the dataframe has a mix of strings and "
    "numbers, in which case you may want to convert the strings in that column to NaN. "
)
MISSING_TRACING_DEPS_ERROR_MSG = (
    "Could not import necessary packages for logging LLM spans "
    "and/or evaluations. It is possible that you are missing the "
    "tracing extra dependencies. If so, please install them with "
    "pip install 'arize[Tracing]'"
)


class Client:
    """
    Arize API Client to log predictions and actuals to the Arize platform from
    pandas.DataFrames
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        space_id: Optional[str] = None,
        space_key: Optional[str] = None,
        uri: Optional[str] = "https://api.arize.com/v1",
        additional_headers: Optional[Dict[str, str]] = None,
        request_verify: Union[bool, str] = True,
    ) -> None:
        """
        Initializes the Arize Client

        Arguments:
        ----------
            api_key (str): Arize provided API key associated with your account. Located on the
                space settings page.
            space_id (str): Arize provided space identifier to connect records to spaces. Located on
                the space settings page.
            space_key (str): [Deprecated] Arize provided identifier to connect records to spaces.
                Located on the space settings.
            uri (str, optional): URI endpoint to send your records to Arize AI. Defaults to
                "https://api.arize.com/v1".
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
        self._files_uri = uri + "/pandas_arrow"
        self._headers = {
            "authorization": self._api_key,
            "space": self._space_key,
            "space_id": self._space_id,
            "sdk-language": "python",
            "language-version": get_python_version(),
            "sdk-version": __version__,
            "sync": "0",  # Defaults to async logging
        }
        if additional_headers is not None:
            reserved_headers = set(self._headers.keys())
            # The header 'schema' is updated in the _post_file method
            reserved_headers.add("schema")
            conflicting_keys = reserved_headers & additional_headers.keys()
            if conflicting_keys:
                raise err.InvalidAdditionalHeaders(conflicting_keys)
            self._headers.update(additional_headers)

    def log_spans(
        self,
        dataframe: pd.DataFrame,
        model_id: str,
        model_version: Optional[str] = None,
        evals_dataframe: Optional[pd.DataFrame] = None,
        datetime_format: str = DEFAULT_DATETIME_FMT,
        validate: Optional[bool] = True,
        path: Optional[str] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = False,
    ) -> requests.Response:
        """
        Logs a pandas dataframe containing LLM tracing data to Arize via a POST request. Returns a
        :class:`Response` object from the Requests HTTP library to ensure successful delivery of
        records.

        Arguments:
        ----------
            dataframe (pd.DataFrame): The dataframe containing the LLM traces.
            model_id (str): A unique name to identify your model in the Arize platform.
            model_version (str, optional): Used to group a subset of traces a given
                model_id to compare and track changes. Defaults to None.
            evals_dataframe (pd.DataFrame, optional): A dataframe containing LLM evaluations data.
                The evaluations are joined to their corresponding spans via a left outer join, i.e.,
                using only `context.span_id` from the spans dataframe. Defaults to None.
            datetime_format (str): format for the timestamp captured in the LLM traces.
                Defaults to "%Y-%m-%dT%H:%M:%S.%f+00:00".
            validate (bool, optional): When set to True, validation is run before sending data.
                Defaults to True.
            path (str, optional): Temporary directory/file to store the serialized data in binary
                before sending to Arize.
            timeout (float, optional): You can stop waiting for a response after a given number
                of seconds with the timeout parameter. Defaults to None.
            verbose: (bool, optional) = When set to true, info messages are printed. Defaults to
                False.

        Returns:
        --------
            `Response` object
        """

        try:
            from .tracing.columns import (
                EVAL_COLUMN_PATTERN,
                ROOT_LEVEL_SPAN_KIND_COL,
                SPAN_KIND_COL,
                SPAN_OPENINFERENCE_COLUMNS,
                SPAN_SPAN_ID_COL,
            )
            from .tracing.utils import convert_timestamps, jsonify_dictionaries
            from .tracing.validation.evals import evals_validation
            from .tracing.validation.spans import spans_validation
        except ImportError:
            raise ImportError(MISSING_TRACING_DEPS_ERROR_MSG)
        # We need our own copy since we will manipulate the underlying data and
        # do not want side effects
        spans_df = dataframe.copy()
        evals_df = None
        if evals_dataframe is not None:
            evals_df = evals_dataframe.copy()

        # Send the number of rows in the dataframe as a header
        # This helps the Arize server to return appropriate feedback, specially for async logging
        self._headers.update({"number-of-rows": str(len(spans_df))})

        # We expect the index to be 0,1,2,3..., len(df)-1. Phoenix, for example, will give us a dataframe
        # with context_id as the index
        reset_dataframe_index(dataframe=spans_df)
        if evals_df is not None:
            reset_dataframe_index(dataframe=evals_df)

        if verbose:
            logger.info("Performing direct input type validation.")
        errors = spans_validation.validate_argument_types(
            spans_dataframe=spans_df,
            model_id=model_id,
            model_version=model_version,
            dt_fmt=datetime_format,
        )
        if evals_df is not None:
            eval_errors = evals_validation.validate_argument_types(
                evals_dataframe=evals_df,
                model_id=model_id,
                model_version=model_version,
            )
            errors += eval_errors
        if errors:
            for e in errors:
                logger.error(e)
            raise err.ValidationFailure(errors)

        if validate:
            if verbose:
                logger.info("Performing dataframe form validation.")
            errors = spans_validation.validate_dataframe_form(spans_dataframe=spans_df)
            if evals_df is not None:
                eval_errors = evals_validation.validate_dataframe_form(evals_dataframe=evals_df)
                errors += eval_errors
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

        if verbose:
            logger.debug("Removing unnecessary columns.")
        spans_df = self._remove_extraneous_columns(
            df=spans_df, column_list=[col.name for col in SPAN_OPENINFERENCE_COLUMNS]
        )
        if evals_df is not None:
            evals_df = self._remove_extraneous_columns(
                df=evals_df,
                column_list=[SPAN_SPAN_ID_COL.name],
                regex=EVAL_COLUMN_PATTERN,
            )

        if model_id is not None:
            model_id = str(model_id)

        if model_version is not None:
            model_version = str(model_version)

        if verbose:
            logger.debug("Converting timestamps.")
        spans_df = convert_timestamps(df=spans_df, fmt=datetime_format)

        if validate:
            if verbose:
                logger.info("Performing values validation.")
            errors = spans_validation.validate_values(
                spans_dataframe=spans_df,
                model_id=model_id,
                model_version=model_version,
            )
            if evals_df is not None:
                eval_errors = evals_validation.validate_values(
                    evals_dataframe=evals_df,
                    model_id=model_id,
                    model_version=model_version,
                )
                errors += eval_errors
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

        if verbose:
            logger.debug("Converting dictionaries to JSON objects.")
        spans_df = jsonify_dictionaries(spans_df)
        if (
            ROOT_LEVEL_SPAN_KIND_COL.name in spans_df.columns
            and SPAN_KIND_COL.name not in spans_df.columns
        ):
            if verbose:
                logger.debug("Moving span kind to atributes")
            spans_df.rename(
                columns={ROOT_LEVEL_SPAN_KIND_COL.name: SPAN_KIND_COL.name},
                inplace=True,
            )

        if evals_df is None:
            df = spans_df
        else:
            # We have already validated that the dataframes both contain the span_id and ensured that
            # they contain no other overlapping columns by removing columns that do not fit their
            # respective conventions.
            if verbose:
                logger.debug("Merging evals and spans dataframes")
            df = pd.merge(spans_df, evals_df, on=SPAN_SPAN_ID_COL.name, how="left")
        try:
            if verbose:
                logger.debug("Getting pyarrow schema from pandas dataframe.")
            pa_schema = pa.Schema.from_pandas(df)
        except pa.ArrowInvalid:
            logger.error(INVALID_ARROW_CONVERSION_MSG)
            raise

        if verbose:
            logger.debug("Getting pyarrow table from pandas dataframe.")
        ta = pa.Table.from_pandas(df)

        proto_schema = proto._get_pb_schema_tracing(
            model_id=model_id,
            model_version=model_version,
        )
        return self._log_arrow(
            pa_table=ta,
            pa_schema=pa_schema,
            proto_schema=proto_schema,
            path=path,
            verbose=verbose,
            timeout=timeout,
        )

    def log_evaluations(
        self,
        dataframe: pd.DataFrame,
        model_id: str,
        model_version: Optional[str] = None,
        validate: Optional[bool] = True,
        path: Optional[str] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = False,
    ) -> requests.Response:
        """
        Logs a pandas dataframe containing LLM evaluations data to Arize via a POST request. The dataframe
        must contain a column `context.span_id` such that Arize can assign each evaluation to its
        respective span.
        Returns a :class:`Response` object from the Requests HTTP library to ensure successful delivery of
        records.

        Arguments:
        ----------
            dataframe (pd.DataFrame): A dataframe containing LLM evaluations data.
            model_id (str): A unique name to identify your model in the Arize platform. It should match
                the model_id of the spans sent previously, to which evaluations will be assigned.
            model_version (str, optional): Used to group a subset of traces a given
                model_id to compare and track changes. It should match the model_id of the spans
                sent previously, to which evaluations will be assigned. Defaults to None.
            validate (bool, optional): When set to True, validation is run before sending data.
                Defaults to True.
            path (str, optional): Temporary directory/file to store the serialized data in binary
                before sending to Arize.
            timeout (float, optional): You can stop waiting for a response after a given number
                of seconds with the timeout parameter. Defaults to None.
            verbose: (bool, optional) = When set to true, info messages are printed. Defaults to
                False.

        Returns:
        --------
            `Response` object
        """

        try:
            from .tracing.columns import EVAL_COLUMN_PATTERN, SPAN_SPAN_ID_COL
            from .tracing.validation.evals import evals_validation
        except ImportError:
            raise ImportError(MISSING_TRACING_DEPS_ERROR_MSG)
        # We need our own copy since we will manipulate the underlying data and
        # do not want side effects
        evals_df = dataframe.copy()

        # Send the number of rows in the dataframe as a header
        # This helps the Arize server to return appropriate feedback, specially for async logging
        self._headers.update({"number-of-rows": str(len(evals_df))})

        # We expect the index to be 0,1,2,3..., len(df)-1. Phoenix, for example, will give us a dataframe
        # with context_id as the index
        reset_dataframe_index(dataframe=evals_df)

        if verbose:
            logger.info("Performing direct input type validation.")
        errors = evals_validation.validate_argument_types(
            evals_dataframe=evals_df,
            model_id=model_id,
            model_version=model_version,
        )
        if errors:
            for e in errors:
                logger.error(e)
            raise err.ValidationFailure(errors)

        if validate:
            if verbose:
                logger.info("Performing dataframe form validation.")
            errors = evals_validation.validate_dataframe_form(evals_dataframe=evals_df)
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

        if verbose:
            logger.debug("Removing unnecessary columns.")
        evals_df = self._remove_extraneous_columns(
            df=evals_df,
            column_list=[SPAN_SPAN_ID_COL.name],
            regex=EVAL_COLUMN_PATTERN,
        )

        if model_id is not None:
            model_id = str(model_id)

        if model_version is not None:
            model_version = str(model_version)

        if validate:
            if verbose:
                logger.info("Performing values validation.")
            errors = evals_validation.validate_values(
                evals_dataframe=evals_df,
                model_id=model_id,
                model_version=model_version,
            )
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

        try:
            if verbose:
                logger.debug("Getting pyarrow schema from pandas dataframe.")
            pa_schema = pa.Schema.from_pandas(evals_df)
        except pa.ArrowInvalid:
            logger.error(INVALID_ARROW_CONVERSION_MSG)
            raise

        if verbose:
            logger.debug("Getting pyarrow table from pandas dataframe.")
        ta = pa.Table.from_pandas(evals_df)

        proto_schema = proto._get_pb_schema_tracing(
            model_id=model_id,
            model_version=model_version,
        )
        return self._log_arrow(
            pa_table=ta,
            pa_schema=pa_schema,
            proto_schema=proto_schema,
            path=path,
            verbose=verbose,
            timeout=timeout,
        )

    def log(
        self,
        dataframe: pd.DataFrame,
        schema: BaseSchema,
        environment: Environments,
        model_id: str,
        model_type: ModelTypes,
        metrics_validation: Optional[List[Metrics]] = None,
        model_version: Optional[str] = None,
        batch_id: Optional[str] = None,
        sync: Optional[bool] = False,
        validate: Optional[bool] = True,
        path: Optional[str] = None,
        surrogate_explainability: Optional[bool] = False,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = False,
    ) -> requests.Response:
        """
        Logs a pandas dataframe containing inferences to Arize via a POST request. Returns a
        :class:`Response` object from the Requests HTTP library to ensure successful delivery of
        records.

        Arguments:
        ----------
            dataframe (pd.DataFrame): The dataframe containing model data.
            schema (BaseSchema): A BaseSchema instance that specifies the column names for corresponding
                data in the dataframe. Can be either a Schema or CorpusSchema (if the environment is
                Environments.CORPUS) object. To use the casting feature, set Schema feature or tag columns
                to a TypedColumns object.
            environment (Environments): The environment the data corresponds to (Production,
                Training, Validation).
            model_id (str): A unique name to identify your model in the Arize platform.
            model_type (ModelTypes): Declare your model type. Can check the supported model types
                running `ModelTypes.list_types()`.
            metrics_validation (List[Metrics], optional): A list of desired metric types;
                defaults to None. When populated, and if validate=True, the presence of schema columns are
                validated against the desired metrics.
            model_version (str, optional): Used to group a subset of predictions and actuals for
                a given model_id to compare and track changes. Defaults to None.
            batch_id (str, optional): Used to distinguish different batch of data under the same
                model_id and  model_version. Defaults to None.
            sync (bool, optional): When sync is set to True, the log call will block, or wait,
                until the data has been successfully ingested by the platform and immediately return the
                status of the log. Defaults to False.
            validate (bool, optional): When set to True, validation is run before sending data.
                Defaults to True.
            path (str, optional): Temporary directory/file to store the serialized data in binary
                before sending to Arize.
            surrogate_explainability (bool, optional): Computes feature importance values using
                the surrogate explainability method. This requires that the arize module is installed with
                the [MimicExplainer] option. If feature importance values are already specified by the
                shap_values_column_names attribute in the Schema, this module will not run. Defaults to
                False.
            timeout (float, optional): You can stop waiting for a response after a given number
                of seconds with the timeout parameter. Defaults to None.
            verbose: (bool, optional) = When set to true, info messages are printed. Defaults to
                False.

        Returns:
        --------
            `Response` object
        """

        # Send the number of rows in the dataframe as a header
        # This helps the Arize server to return appropriate feedback, specially for async logging
        self._headers.update({"number-of-rows": str(len(dataframe))})
        # Deep copy the schema since we might modify it to add certain columns and don't
        # want to cause side effects
        schema = copy.deepcopy(schema)

        # If typed columns are specified in the schema,
        # apply casting and return new copies of the dataframe + schema.
        # All downstream validations are kept the same.
        # note: we don't do any casting for Corpus schemas.
        if isinstance(schema, Schema) and schema.has_typed_columns():
            # The pandas nullable string column type (StringDType) is still considered experimental
            # and is unavailable before pandas 1.0.0.
            # Thus we can only offer this functionality with pandas>=1.0.0.
            # TODO (Hannah): After we remove support for 0.25.3, remove this check.
            pandas_version = parse_version(pd.__version__)
            if pandas_version < parse_version(ETL_MINIMUM_PANDAS_VERSION):
                raise ImportError(ETL_ERROR_MESSAGE)
            try:
                dataframe, schema = cast_typed_columns(dataframe, schema)
            except Exception as e:
                logger.error(e)
                raise

        # Warning for when prediction_label is not provided and we generate default prediction
        # labels for GENERATIVE_LLM models
        if model_type == ModelTypes.GENERATIVE_LLM and environment != Environments.CORPUS:
            self._generative_model_warnings(dataframe, schema)

        if verbose:
            logger.info("Performing required validation.")
        errors = Validator.validate_required_checks(
            dataframe=dataframe,
            model_id=model_id,
            environment=environment,
            schema=schema,
            model_version=model_version,
            batch_id=batch_id,
        )
        if errors:
            for e in errors:
                logger.error(e)
            raise err.ValidationFailure(errors)

        if model_id is not None:
            model_id = str(model_id)

        if model_version is not None:
            model_version = str(model_version)

        if batch_id is not None:
            batch_id = str(batch_id)

        if validate:
            if verbose:
                logger.info("Performing parameters validation.")
            errors = Validator.validate_params(
                dataframe=dataframe,
                model_id=model_id,
                model_type=model_type,
                environment=environment,
                schema=schema,
                metric_families=metrics_validation,
                model_version=model_version,
                batch_id=batch_id,
            )
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

        if verbose:
            logger.debug("Removing unnecessary columns.")
        dataframe = self._remove_extraneous_columns(df=dataframe, schema=schema)

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
                logger.info("Running surrogate_explainability.")

            if is_python_version_below_required_min(
                min_req_version=SURROGATE_EXPLAINER_MIN_PYTHON_VERSION
            ):
                raise RuntimeError(
                    "Cannot use Arize's Surrogate Explainer. "
                    f"Minimum Python version required is {SURROGATE_EXPLAINER_MIN_PYTHON_VERSION} and "
                    f"{get_python_version()} was found instead"
                )
            try:
                from arize.pandas.surrogate_explainer.mimic import Mimic
            except ImportError:
                raise ImportError(
                    "To enable surrogate explainability, "
                    "the arize module must be installed with the MimicExplainer option: pip "
                    "install 'arize[MimicExplainer]'."
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
                logger.debug("Getting pyarrow schema from pandas dataframe.")
            # TODO: Addition of column for GENERATIVE models should occur at the
            # beginning of the log function, so validations are applied to the resulting schema
            if model_type == ModelTypes.GENERATIVE_LLM and environment != Environments.CORPUS:
                dataframe, schema = self._add_generative_llm_columns(dataframe, schema)

            pa_schema = pa.Schema.from_pandas(dataframe)
        except pa.ArrowInvalid:
            logger.error(INVALID_ARROW_CONVERSION_MSG)
            raise

        if validate:
            if verbose:
                logger.info("Performing types validation.")
            errors = Validator.validate_types(
                model_type=model_type,
                schema=schema,
                pyarrow_schema=pa_schema,
            )
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)
        if validate:
            if verbose:
                logger.info("Performing values validation.")
            errors = Validator.validate_values(
                dataframe=dataframe,
                environment=environment,
                schema=schema,
                model_type=model_type,
            )
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

        if verbose:
            logger.debug("Getting pyarrow table from pandas dataframe.")
        ta = pa.Table.from_pandas(dataframe)

        if environment == Environments.CORPUS:
            proto_schema = proto._get_pb_schema_corpus(schema, model_id, model_type, environment)
        else:
            proto_schema = proto._get_pb_schema(
                schema, model_id, model_version, model_type, environment, batch_id
            )

        if isinstance(schema, Schema) and not schema.has_prediction_columns():
            logger.warning(
                "Logging actuals without any predictions may result in "
                "unexpected behavior if corresponding predictions have not been logged prior. "
                "Please see the docs at https://docs.arize.com/arize/sending-data/sending-data-faq"
                "#what-happens-after-i-send-in-actual-data"
            )

        return self._log_arrow(
            pa_table=ta,
            pa_schema=pa_schema,
            proto_schema=proto_schema,
            path=path,
            sync=sync,
            verbose=verbose,
            timeout=timeout,
        )

    def _log_arrow(
        self,
        pa_table: pa.Table,
        pa_schema: pa.Schema,
        proto_schema: pb2.Schema,
        path: Optional[str] = None,
        sync: bool = False,
        verbose: bool = False,
        timeout: Optional[float] = None,
    ) -> requests.Response:
        if verbose:
            logger.debug("Serializing schema.")
        base64_schema = base64.b64encode(proto_schema.SerializeToString())
        pa_schema = self._append_to_pyarrow_metadata(pa_schema, {"arize-schema": base64_schema})

        if path is None:
            tmp_dir = tempfile.mkdtemp()
            fd, tmp_file = tempfile.mkstemp(dir=tmp_dir)
            # This way of handling temp files is not ideal, but necessary for it to work
            # for Windows machines. A Windows corner case is on exiting a tempfile context manager,
            # PermissionError can be thrown if non-writable files are placed into a
            # tempfile.TemporaryDirectory. Python 3.10 fixed this issue by adding argument
            # TemporaryDirectory(ignore_cleanup_errors=True). See code that will work well across
            # operating systems: https://www.scivision.dev/python-tempfile-permission-error-windows/
        else:
            tmp_file = path

        try:
            if verbose:
                logger.debug(f"Writing table to temporary file: {tmp_file}")
            writer = pa.ipc.RecordBatchStreamWriter(tmp_file, pa_schema)
            writer.write_table(pa_table, max_chunksize=65536)
            writer.close()
            if verbose:
                logger.info("Sending file to Arize")
            response = self._post_file(
                path=tmp_file,
                sync=sync,
                timeout=timeout,
            )
        finally:
            if path is None:
                # NOTE: This try-catch should also be updated/removed when
                # Python >=3.10 is required, see comment above
                try:
                    os.close(fd)
                    shutil.rmtree(tmp_dir)
                except PermissionError:
                    pass

        try:
            url = reconstruct_url(response, drop_in_data_ingestion=False)
            if url != "":
                logger.info(f"Success! Check out your data at {url}")
        except Exception:
            pass

        return response

    def _post_file(
        self,
        path: str,
        sync: bool = False,
        timeout: Optional[float] = None,
    ) -> requests.Response:
        self._headers.update({"sync": "1" if sync is True else "0"})
        with open(path, "rb") as f:
            return requests.post(
                self._files_uri,
                timeout=timeout,
                data=f,
                headers=self._headers,
                verify=self._request_verify,
            )

    @staticmethod
    def _add_default_prediction_label_column(df: pd.DataFrame) -> pd.DataFrame:
        df.insert(loc=0, column=GENERATED_PREDICTION_LABEL_COL, value=1, allow_duplicates=False)
        return df

    # Add in all the relevant columns for generative LLMs and modify the schema accordingly
    def _add_generative_llm_columns(
        self, dataframe: pd.DataFrame, schema: Schema
    ) -> (pd.DataFrame, Schema):
        if schema.prediction_label_column_name is None and schema.actual_label_column_name is None:
            dataframe = self._add_default_prediction_label_column(dataframe)
            schema = schema.replace(prediction_label_column_name=GENERATED_PREDICTION_LABEL_COL)
        if (
            schema.llm_config_column_names and schema.llm_config_column_names.params_column_name
        ) is not None:
            dataframe = self._add_json_llm_params_column(
                dataframe, schema.llm_config_column_names.params_column_name
            )
            schema = schema.replace(
                llm_config_column_names=LLMConfigColumnNames(
                    model_column_name=schema.llm_config_column_names.model_column_name,
                    params_column_name=GENERATED_LLM_PARAMS_JSON_COL,
                )
            )
        if schema.llm_run_metadata_column_names is not None:
            if schema.llm_run_metadata_column_names.total_token_count_column_name is not None:
                dataframe = self._add_reserved_tag_column(
                    dataframe,
                    LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
                    schema.llm_run_metadata_column_names.total_token_count_column_name,
                )
                schema.tag_column_names.append(LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME)
            if schema.llm_run_metadata_column_names.prompt_token_count_column_name is not None:
                dataframe = self._add_reserved_tag_column(
                    dataframe,
                    LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
                    schema.llm_run_metadata_column_names.prompt_token_count_column_name,
                )
                schema.tag_column_names.append(LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME)
            if schema.llm_run_metadata_column_names.response_token_count_column_name is not None:
                dataframe = self._add_reserved_tag_column(
                    dataframe,
                    LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
                    schema.llm_run_metadata_column_names.response_token_count_column_name,
                )
                schema.tag_column_names.append(LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME)
            if schema.llm_run_metadata_column_names.response_latency_ms_column_name is not None:
                dataframe = self._add_reserved_tag_column(
                    dataframe,
                    LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
                    schema.llm_run_metadata_column_names.response_latency_ms_column_name,
                )
                schema.tag_column_names.append(LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME)
        return dataframe, schema

    @staticmethod
    def _add_json_llm_params_column(df: pd.DataFrame, llm_params_col_name: str) -> pd.DataFrame:
        df[GENERATED_LLM_PARAMS_JSON_COL] = df[llm_params_col_name].apply(
            lambda d: json.dumps(d, indent=4)
        )
        return df

    # Adds in a new column to the dataframe under the name of reserved_tag_column_name that is a
    # copy of the tag_column_name column
    @staticmethod
    def _add_reserved_tag_column(
        df: pd.DataFrame, reserved_tag_column_name: str, tag_column_name: str
    ) -> pd.DataFrame:
        if reserved_tag_column_name == tag_column_name:
            return df
        df[reserved_tag_column_name] = df[tag_column_name]
        return df

    # Returns the dataframe with only the columns that are in the schema, in the column_list, or
    # match the regex
    @staticmethod
    def _remove_extraneous_columns(
        df: pd.DataFrame,
        schema: Optional[BaseSchema] = None,
        column_list: Optional[List[str]] = None,
        regex: Optional[str] = None,
    ) -> pd.DataFrame:
        relevant_columns = set()
        if schema is not None:
            relevant_columns.update(schema.get_used_columns())
        if column_list is not None:
            relevant_columns.update(column_list)
        if regex is not None:
            relevant_columns.update([col for col in df.columns if re.match(regex, col)])
        final_columns = list(set(df.columns) & relevant_columns)
        return df[final_columns]

    @staticmethod
    def _generative_model_warnings(df: pd.DataFrame, schema: Schema):
        # Warning for when prediction_label_column_name is not provided
        if schema.prediction_label_column_name is None and schema.actual_label_column_name is None:
            # Warning for when actual_label is also not provided
            logger.warning(
                "prediction_label_column_name and actual_label_column_name were both not provided, "
                "so a default prediction label equal to 1 will be set on this GENERATIVE_LLM model data."
            )
        if schema.actual_label_column_name is None:
            logger.warning(
                "actual_label_column_name was not provided. Some metrics that require actual labels, "
                "e.g. correctness or accuracy, may not be computed."
            )

    @staticmethod
    def _append_to_pyarrow_metadata(pa_schema: pa.Schema, new_metadata: Dict[str, Any]):
        metadata = pa_schema.metadata
        conflicting_keys = metadata.keys() & new_metadata.keys()
        if conflicting_keys:
            raise KeyError(
                "Cannot append metadata to pyarrow schema. "
                f"There are conflicting keys: {log_a_list(conflicting_keys, join_word='and')}"
            )
        metadata.update(new_metadata)
        return pa_schema.with_metadata(metadata)
