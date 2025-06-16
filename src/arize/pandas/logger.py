# type: ignore[pb2]
import base64
import copy
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pyarrow as pa
import requests
from google.protobuf import json_format
from packaging.version import parse as parse_version
from pyarrow import flight

from arize.pandas.proto.requests_pb2 import (
    DoPutRequest,
    WriteSpanAnnotationRequest,
    WriteSpanAnnotationResponse,
    WriteSpanAttributesMetadataRequest,
    WriteSpanAttributesMetadataResponse,
    WriteSpanEvaluationRequest,
    WriteSpanEvaluationResponse,
)
from arize.version import __version__

from .. import public_pb2 as pb2
from ..utils import proto
from ..utils.constants import (
    API_KEY_ENVVAR_NAME,
    DEFAULT_ARIZE_FLIGHT_HOST,
    DEFAULT_ARIZE_FLIGHT_PORT,
    DEFAULT_TRANSPORT_SCHEME,
    DEVELOPER_KEY_ENVVAR_NAME,
    GENERATED_LLM_PARAMS_JSON_COL,
    GENERATED_PREDICTION_LABEL_COL,
    LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
    SPACE_ID_ENVVAR_NAME,
    SPACE_KEY_ENVVAR_NAME,
)
from ..utils.errors import AuthError, InvalidCertificateFile
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
from .etl.casting import (
    ETL_ERROR_MESSAGE,
    ETL_MINIMUM_PANDAS_VERSION,
    cast_typed_columns,
)
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
    Arize API Client
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        space_id: Optional[str] = None,
        space_key: Optional[str] = None,
        uri: Optional[str] = "https://api.arize.com/v1",
        additional_headers: Optional[Dict[str, str]] = None,
        request_verify: Union[bool, str] = True,
        developer_key: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        scheme: str = DEFAULT_TRANSPORT_SCHEME,
    ) -> None:
        """
        Initializes the Arize Client

        Args:
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
            developer_key (str): [Deprecated] You only need the api_key for all data logging operations.
            host (str, optional): Arize Flight server host. Defaults to DEFAULT_ARIZE_FLIGHT_HOST.
            port (int, optional): Arize Flight server port. Defaults to DEFAULT_ARIZE_FLIGHT_PORT.
            scheme (str, optional): Arize Flight server scheme. Defaults to DEFAULT_TRANSPORT_SCHEME.
        """
        self._api_key = api_key or os.getenv(API_KEY_ENVVAR_NAME)
        self._space_id = space_id or os.getenv(SPACE_ID_ENVVAR_NAME)
        self._space_key = space_key or os.getenv(SPACE_KEY_ENVVAR_NAME)
        self._developer_key = developer_key or os.getenv(
            DEVELOPER_KEY_ENVVAR_NAME
        )
        if self._space_key is not None:
            logger.warning(
                "The space_key parameter is deprecated and will be removed in a future release. "
                "Please use the space_id parameter instead."
            )
        if self._developer_key is not None:
            logger.warning(
                "The developer_key parameter is deprecated and will be removed in a future release. "
                "You only need the api_key for all data logging operations."
            )
        else:
            self._developer_key = self._api_key

        if isinstance(request_verify, str) and not os.path.isfile(
            request_verify
        ):
            raise InvalidCertificateFile(request_verify)
        self._request_verify = request_verify
        self._files_uri = uri + "/pandas_arrow"
        self._headers = {
            "authorization": self._api_key,
            "arize-space-id": self._space_id,
            "space_id": self._space_id,  # deprecated, will remove in future release
            "space": self._space_key,  # deprecated, will remove in future release
            "arize-space-key": self._space_key,  # deprecated, will remove in future release
            "arize-interface": "batch",
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

        # required for sending events to Flight server
        self._host = host if host else DEFAULT_ARIZE_FLIGHT_HOST
        self._port = port if port else DEFAULT_ARIZE_FLIGHT_PORT
        self._scheme = scheme
        # Only initialize FlightSession if all required params are set
        self._flight_session = None
        has_flight_params = all(
            [self._host, self._port, self._developer_key, self._space_id]
        )
        if has_flight_params:
            self._flight_session = FlightSession(
                host=self._host,
                port=self._port,
                api_key=self._developer_key,
                space_id=self._space_id,
                scheme=self._scheme,
            )

    def log_spans(
        self,
        dataframe: pd.DataFrame,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        evals_dataframe: Optional[pd.DataFrame] = None,
        datetime_format: str = DEFAULT_DATETIME_FMT,
        validate: Optional[bool] = True,
        path: Optional[str] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = False,
        project_name: Optional[str] = None,
    ) -> requests.Response:
        """
        Logs a pandas dataframe containing LLM tracing data to Arize via a POST request. Returns a
        :class:`Response` object from the Requests HTTP library to ensure successful delivery of
        records.

        Args:
            dataframe (pd.DataFrame): The dataframe containing the LLM traces.
            model_id (str): A unique name to identify your model in the Arize platform.
                (Deprecated: Use `project_name` instead.)
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
            project_name (str, optional): A unique name to identify your project in the Arize platform.
                Either model_id or project_name must be provided.

        Returns:
            `Response` object

        """
        # This method requires the API key and either space ID or space key to be set
        # api_key and one of space_id or space_key must be provided
        if not self._api_key or not (self._space_id or self._space_key):
            raise AuthError(
                missing_space_id=not (self._space_id or self._space_key),
                missing_api_key=not self._api_key,
                method_name="log_spans",
            )
        try:
            from .tracing.columns import (
                EVAL_COLUMN_PATTERN,
                ROOT_LEVEL_SPAN_KIND_COL,
                SPAN_KIND_COL,
                SPAN_OPENINFERENCE_COLUMNS,
                SPAN_SPAN_ID_COL,
            )
            from .tracing.utils import (
                convert_timestamps,
                extract_project_name_from_params,
                jsonify_dictionaries,
            )
            from .tracing.validation.evals import evals_validation
            from .tracing.validation.spans import spans_validation
        except ImportError:
            raise ImportError(MISSING_TRACING_DEPS_ERROR_MSG) from None

        project_name = extract_project_name_from_params(
            model_id=model_id, project_name=project_name
        )

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
            project_name=project_name,
            model_version=model_version,
            dt_fmt=datetime_format,
        )
        if evals_df is not None:
            eval_errors = evals_validation.validate_argument_types(
                evals_dataframe=evals_df,
                project_name=project_name,
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
            errors = spans_validation.validate_dataframe_form(
                spans_dataframe=spans_df
            )
            if evals_df is not None:
                eval_errors = evals_validation.validate_dataframe_form(
                    evals_dataframe=evals_df
                )
                errors += eval_errors
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

        if verbose:
            logger.debug("Removing unnecessary columns.")
        spans_df = self._remove_extraneous_columns(
            df=spans_df,
            column_list=[col.name for col in SPAN_OPENINFERENCE_COLUMNS],
        )
        if evals_df is not None:
            evals_df = self._remove_extraneous_columns(
                df=evals_df,
                column_list=[SPAN_SPAN_ID_COL.name],
                regex=EVAL_COLUMN_PATTERN,
            )

        if project_name is not None:
            project_name = str(project_name)

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
                project_name=project_name,
                model_version=model_version,
            )
            if evals_df is not None:
                eval_errors = evals_validation.validate_values(
                    evals_dataframe=evals_df,
                    project_name=project_name,
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
            df = pd.merge(
                spans_df, evals_df, on=SPAN_SPAN_ID_COL.name, how="left"
            )
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
            model_id=project_name,
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
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        validate: Optional[bool] = True,
        path: Optional[str] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = False,
        project_name: Optional[str] = None,
    ) -> requests.Response:
        """
        Logs a pandas dataframe containing LLM evaluations data to Arize via a POST request. The dataframe
        must contain a column `context.span_id` such that Arize can assign each evaluation to its
        respective span.
        Returns a :class:`Response` object from the Requests HTTP library to ensure successful delivery of
        records.

        Args:
            dataframe (pd.DataFrame): A dataframe containing LLM evaluations data.
            model_id (str): A unique name to identify your model in the Arize platform.
                (Deprecated: Use `project_name` instead.)
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
            project_name (str, optional): A unique name to identify your project in the Arize platform.
                Either model_id or project_name must be provided.

        Returns:
            `Response` object

        """
        # This method requires the API key and either space ID or space key to be set
        # api_key and one of space_id or space_key must be provided
        if not self._api_key or not (self._space_id or self._space_key):
            raise AuthError(
                missing_space_id=not (self._space_id or self._space_key),
                missing_api_key=not self._api_key,
                method_name="log_evaluations",
            )
        try:
            from .tracing.columns import EVAL_COLUMN_PATTERN, SPAN_SPAN_ID_COL
            from .tracing.utils import extract_project_name_from_params
            from .tracing.validation.evals import evals_validation
        except ImportError:
            raise ImportError(MISSING_TRACING_DEPS_ERROR_MSG) from None

        project_name = extract_project_name_from_params(project_name, model_id)

        # We need our own copy since we will manipulate the underlying data and
        # do not want side effects
        evals_df = dataframe.copy()

        # Send the number of rows in the dataframe as a header
        # This helps the Arize server to return appropriate feedback, specially for async logging
        self._headers.update({"number-of-rows": str(len(evals_df))})

        # We expect the index to be 0,1,2,3..., len(df)-1. Phoenix, for example, will give us a dataframe
        # with context_id as the index; the old index is not meaningful in our copy of the original dataframe
        # so we can drop it.
        evals_df.reset_index(inplace=True, drop=True)

        if verbose:
            logger.info("Performing direct input type validation.")
        errors = evals_validation.validate_argument_types(
            evals_dataframe=evals_df,
            project_name=project_name,
            model_version=model_version,
        )
        if errors:
            for e in errors:
                logger.error(e)
            raise err.ValidationFailure(errors)

        if validate:
            if verbose:
                logger.info("Performing dataframe form validation.")
            errors = evals_validation.validate_dataframe_form(
                evals_dataframe=evals_df
            )
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

        if project_name is not None:
            project_name = str(project_name)

        if model_version is not None:
            model_version = str(model_version)

        if validate:
            if verbose:
                logger.info("Performing values validation.")
            errors = evals_validation.validate_values(
                evals_dataframe=evals_df,
                project_name=project_name,
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
            model_id=project_name,
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

    def log_evaluations_sync(
        self,
        dataframe: pd.DataFrame,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        validate: Optional[bool] = True,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = False,
        project_name: Optional[str] = None,
    ) -> WriteSpanEvaluationResponse:
        """
        Logs a pandas dataframe containing LLM evaluations data to Arize via a Flight gRPC request.
        The dataframe must contain a column `context.span_id`
        such that Arize can assign each evaluation to its respective span.

        Args:
            dataframe (pd.DataFrame): A dataframe containing LLM evaluations data.
            model_id (str): A unique name to identify your model in the Arize platform.
                (Deprecated: Use `project_name` instead.)
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
            project_name (str, optional): A unique name to identify your project in the Arize platform.
                Either model_id or project_name must be provided.
        """
        # This method requires the space ID and developer key to be set
        if not self._space_id or not self._developer_key:
            raise AuthError(
                missing_space_id=not self._space_id,
                missing_developer_key=not self._developer_key,
                method_name="log_evaluations_sync",
            )
        try:
            from .tracing.columns import EVAL_COLUMN_PATTERN, SPAN_SPAN_ID_COL
            from .tracing.utils import extract_project_name_from_params
            from .tracing.validation.evals import evals_validation
        except ImportError:
            raise ImportError(MISSING_TRACING_DEPS_ERROR_MSG) from None

        project_name = extract_project_name_from_params(project_name, model_id)

        # We need our own copy since we will manipulate the underlying data and
        # do not want side effects
        evals_df = dataframe.copy()

        # Send the number of rows in the dataframe as a header
        # This helps the Arize server to return appropriate feedback, specially for async logging
        self._headers.update({"number-of-rows": str(len(evals_df))})

        # We expect the index to be 0,1,2,3..., len(df)-1. Phoenix, for example, will give us a dataframe
        # with context_id as the index; the old index is not meaningful in our copy of the original dataframe
        # so we can drop it.
        evals_df.reset_index(inplace=True, drop=True)

        if verbose:
            logger.info("Performing direct input type validation.")
        errors = evals_validation.validate_argument_types(
            evals_dataframe=evals_df,
            project_name=project_name,
            model_version=model_version,
        )
        if errors:
            for e in errors:
                logger.error(e)
            raise err.ValidationFailure(errors)

        if validate:
            if verbose:
                logger.info("Performing dataframe form validation.")
            errors = evals_validation.validate_dataframe_form(
                evals_dataframe=evals_df
            )
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

        if project_name is not None:
            project_name = str(project_name)

        if model_version is not None:
            model_version = str(model_version)

        if validate:
            if verbose:
                logger.info("Performing values validation.")
            errors = evals_validation.validate_values(
                evals_dataframe=evals_df,
                project_name=project_name,
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
            model_id=project_name,
            model_version=model_version,
        )
        return self._log_arrow_flight(
            pa_table=ta,
            pa_schema=pa_schema,
            proto_schema=proto_schema,
            verbose=verbose,
            model_id=project_name,
            model_version=model_version,
            request_type="evaluation",
        )

    def log_annotations(
        self,
        dataframe: pd.DataFrame,
        project_name: str,
        validate: bool = True,
        verbose: bool = False,
    ) -> WriteSpanAnnotationResponse:
        """
        Logs a pandas dataframe containing LLM span annotations to Arize via a Flight gRPC request.
        The dataframe must contain a column `context.span_id`
        such that Arize can assign each annotation to its respective span.
        Annotation columns should follow the pattern `annotation.<name>.<suffix>` where suffix is
        either `label` or `score`. An optional `annotation.notes` column can be included for
        free-form text notes.

        Args:
            dataframe (pd.DataFrame): A dataframe containing LLM annotation data.
            project_name (str): A unique name to identify your project in the Arize platform.
            validate (bool, optional): When set to True, validation is run before sending data.
                Defaults to True.
            verbose: (bool, optional) = When set to true, info messages are printed. Defaults to
                False.
        """
        # Auth check
        if not self._space_id or not self._developer_key:
            raise AuthError(
                missing_space_id=not self._space_id,
                missing_developer_key=not self._developer_key,
                method_name="log_annotations",
            )
        try:
            # Import validation and required columns locally
            from .tracing.columns import (
                ANNOTATION_COLUMN_PATTERN,
                ANNOTATION_NOTES_COLUMN_NAME,
                SPAN_SPAN_ID_COL,
            )
            from .tracing.validation.annotations import annotations_validation
        except ImportError:
            raise ImportError(MISSING_TRACING_DEPS_ERROR_MSG) from None

        anno_df = dataframe.copy()
        self._headers.update({"number-of-rows": str(len(anno_df))})
        anno_df.reset_index(inplace=True, drop=True)

        if verbose:
            logger.info(
                "Checking for and autogenerating missing updated_by/updated_at annotation columns."
            )
        try:
            import re
            from datetime import datetime, timezone

            from .tracing.columns import (
                ANNOTATION_COLUMN_PATTERN,
                ANNOTATION_LABEL_SUFFIX,
                ANNOTATION_SCORE_SUFFIX,
                ANNOTATION_UPDATED_AT_SUFFIX,
                ANNOTATION_UPDATED_BY_SUFFIX,
            )

            annotation_cols = [
                col
                for col in anno_df.columns
                if re.match(ANNOTATION_COLUMN_PATTERN, col)
            ]
            annotation_names = set()
            # Extract unique annotation names (e.g., "quality" from "annotation.quality.label")
            for col in annotation_cols:
                match = re.match(r"^annotation\.([a-zA-Z0-9_\s]+?)(\..+)$", col)
                if match:
                    annotation_names.add(match.group(1))

            if verbose:
                logger.info(f"Found annotation names: {annotation_names}")

            # Timestamp in nanoseconds
            current_time_ns = int(
                datetime.now(timezone.utc).timestamp() * 1_000_000_000
            )

            for name in annotation_names:
                updated_by_col = (
                    f"annotation.{name}{ANNOTATION_UPDATED_BY_SUFFIX}"
                )
                updated_at_col = (
                    f"annotation.{name}{ANNOTATION_UPDATED_AT_SUFFIX}"
                )
                label_col = f"annotation.{name}{ANNOTATION_LABEL_SUFFIX}"
                score_col = f"annotation.{name}{ANNOTATION_SCORE_SUFFIX}"

                # Check if *any* part of this annotation exists (label or score)
                # Only add metadata if the annotation itself is present
                if label_col in anno_df.columns or score_col in anno_df.columns:
                    if updated_by_col not in anno_df.columns:
                        if verbose:
                            logger.info(
                                f"Autogenerating column: {updated_by_col}"
                            )
                        anno_df[updated_by_col] = "SDK"

                    if updated_at_col not in anno_df.columns:
                        if verbose:
                            logger.info(
                                f"Autogenerating column: {updated_at_col}"
                            )
                        anno_df[updated_at_col] = current_time_ns
                else:
                    if verbose:
                        logger.info(
                            f"Skipping metadata generation for '{name}' as no label or score column found."
                        )

        except Exception as e:
            logger.error(
                f"Error during annotation metadata autogeneration: {e}",
                exc_info=True,
            )

        if ANNOTATION_NOTES_COLUMN_NAME in anno_df.columns:
            if verbose:
                logger.info(
                    f"Formatting {ANNOTATION_NOTES_COLUMN_NAME} column to JSON strings within lists."
                )
            try:
                import json
                from datetime import datetime, timezone

                def _format_note_for_storage(note_text):
                    if pd.isna(note_text):
                        return None
                    note_obj = {
                        "text": str(note_text),
                        "updated_by": "SDK",
                        "updated_at": current_time_ns,
                    }
                    return [json.dumps(note_obj)]

                anno_df[ANNOTATION_NOTES_COLUMN_NAME] = anno_df[
                    ANNOTATION_NOTES_COLUMN_NAME
                ].apply(_format_note_for_storage)
            except Exception as e:
                logger.error(
                    f"Error during annotation notes formatting: {e}",
                    exc_info=True,
                )

        if verbose:
            logger.info(
                "Performing direct input type validation for annotations."
            )
        errors = annotations_validation.validate_argument_types(
            annotations_dataframe=anno_df,
            project_name=project_name,
        )
        if errors:
            for e in errors:
                logger.error(e)
            raise err.ValidationFailure(errors)

        if validate:
            if verbose:
                logger.info(
                    "Performing dataframe form validation for annotations."
                )
            errors = annotations_validation.validate_dataframe_form(
                annotations_dataframe=anno_df
            )
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

        if verbose:
            logger.info("Removing unnecessary annotation columns.")
        # Update columns to keep: span_id, annotation.notes, and annotation pattern
        columns_to_keep = [SPAN_SPAN_ID_COL.name]
        if ANNOTATION_NOTES_COLUMN_NAME in anno_df.columns:
            columns_to_keep.append(ANNOTATION_NOTES_COLUMN_NAME)
        anno_df = self._remove_extraneous_columns(
            df=anno_df,
            column_list=columns_to_keep,
            regex=ANNOTATION_COLUMN_PATTERN,
        )

        if project_name is not None:
            project_name = str(project_name)

        if validate:
            if verbose:
                logger.info("Performing annotation values validation.")

            errors = annotations_validation.validate_values(
                annotations_dataframe=anno_df,
                project_name=project_name,
            )
            if errors:
                for e in errors:
                    logger.error(e)
                raise err.ValidationFailure(errors)

        try:
            if verbose:
                logger.info("Getting pyarrow schema from annotation dataframe.")
            try:
                pa_schema = pa.Schema.from_pandas(anno_df, preserve_index=False)
                if verbose:
                    logger.info(f"Inferred schema: {pa_schema}")
                # Verify the inferred type for notes if the column exists
                if ANNOTATION_NOTES_COLUMN_NAME in anno_df.columns:
                    notes_field = pa_schema.field(ANNOTATION_NOTES_COLUMN_NAME)
                    if not (
                        isinstance(notes_field.type, pa.ListType)
                        and notes_field.type.value_type == pa.string()
                    ):
                        logger.warning(
                            f"Warning: Inferred type for {ANNOTATION_NOTES_COLUMN_NAME} is "
                            f"{notes_field.type}, expected list<string>."
                        )

            except pa.ArrowInvalid as e:
                logger.error(f"Error during schema inference/creation: {e}")
                logger.error(INVALID_ARROW_CONVERSION_MSG)
                raise
            except Exception as e:
                logger.error(f"Unexpected error during schema definition: {e}")
                raise

            if verbose:
                logger.info(
                    "Getting pyarrow table from annotation dataframe using inferred schema."
                )
            try:
                # Create the table using the inferred schema
                ta = pa.Table.from_pandas(
                    anno_df, schema=pa_schema, preserve_index=False
                )
            except Exception as e:
                logger.error(
                    f"Error creating Arrow Table with inferred schema: {e}"
                )
                # Log dataframe details for debugging
                logger.info(f"DataFrame info before error:\\n{anno_df.info()}")
                logger.info(f"DataFrame head:\\n{anno_df.head().to_string()}")
                raise

        except pa.ArrowInvalid:
            logger.error(INVALID_ARROW_CONVERSION_MSG)
            raise

        proto_schema = proto._get_pb_schema_tracing(
            model_id=project_name,
            model_version=None,
        )

        return self._log_arrow_flight(
            pa_table=ta,
            pa_schema=pa_schema,
            proto_schema=proto_schema,
            verbose=verbose,
            model_id=project_name,
            model_version=None,
            request_type="annotation",
        )

    def log_metadata(
        self,
        dataframe: pd.DataFrame,
        project_name: str,
        patch_document_column_name: str = "patch_document",
        validate: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Log metadata updates using JSON Merge Patch format. This method is only supported for LLM model types.

        The dataframe must contain a column `context.span_id` to identify spans and either:
        1. A column with JSON patch documents (specified by patch_document_column_name), or
        2. One or more columns with prefix `attributes.metadata.` that will be automatically
           converted to a patch document (e.g., `attributes.metadata.tag` → `{"tag": value}`).

        If both methods are used, the explicit patch document is applied after the individual field updates.
        The patches will be applied to the `attributes.metadata` field of each span.

        **Type Handling:**
        - The client primarily supports string, integer, and float data types.
        - Boolean values are converted to string representations.
        - Nested JSON objects and arrays are serialized to JSON strings during transmission.
        - Setting a field to `None` or `null` will set the field to JSON null in the metadata.
          Note: This differs from standard JSON Merge Patch where null values remove fields.

        Args:
            dataframe: DataFrame with span_ids and either patch documents or metadata field columns.
            project_name: A unique name to identify your project in the Arize platform.
            patch_document_column_name: Name of the column containing JSON patch documents.
                Defaults to "patch_document".
            validate: When set to True, validation is run before sending data.
            verbose: When set to true, info messages are printed.

        Returns:
            Dictionary containing update results with the following keys:
                - spans_processed: Total number of spans in the input dataframe
                - spans_updated: Count of successfully updated span metadata records
                - spans_failed: Count of spans that failed to update
                - errors: List of dictionaries with 'span_id' and 'error_message' keys for each failed span

            Error types from the server include:
                - parse_failure: Failed to parse JSON metadata
                - patch_failure: Failed to apply JSON patch
                - type_conflict: Type conflict in metadata
                - connection_failure: Connection issues
                - segment_not_found: No matching segment found
                - druid_rejection: Backend rejected the update

        Raises:
            AuthError: When API key or space ID is missing
            ValidationFailure: When validation of the dataframe or values fails
            ImportError: When required tracing dependencies are missing
            ArrowInvalid: When the dataframe cannot be converted to Arrow format
            RuntimeError: If the request fails or no response is received

        Example:
            ```python
            # Method 1: Using a patch document
            df = pd.DataFrame(
                {
                    "context.span_id": ["span1", "span2"],
                    "patch_document": [{"tag": "important"}, {"priority": "high"}],
                }
            )

            # Method 2: Using direct field columns
            df = pd.DataFrame(
                {
                    "context.span_id": ["span1", "span2"],
                    "attributes.metadata.tag": ["important", "standard"],
                    "attributes.metadata.priority": ["high", "medium"],
                }
            )

            # Method 3: Combining both approaches
            df = pd.DataFrame(
                {
                    "context.span_id": ["span1"],
                    "attributes.metadata.tag": ["important"],
                    "patch_document": [
                        {"priority": "high"}
                    ],  # This will override any conflicting fields
                }
            )

            # Method 4: Setting fields to null
            df = pd.DataFrame(
                {
                    "context.span_id": ["span1"],
                    "attributes.metadata.old_field": [
                        None
                    ],  # Sets field to JSON null
                    "patch_document": [
                        {"other_field": None}
                    ],  # Also sets field to JSON null
                }
            )
            ```
        """
        # Auth check
        if not self._space_id or not self._developer_key:
            raise AuthError(
                missing_space_id=not self._space_id,
                missing_developer_key=not self._developer_key,
                method_name="log_metadata",
            )

        try:
            # Import validation modules
            from .tracing.columns import SPAN_SPAN_ID_COL
            from .tracing.validation.metadata.argument_validation import (
                validate_argument_types,
            )
            from .tracing.validation.metadata.dataframe_form_validation import (
                validate_dataframe_form,
            )
            from .tracing.validation.metadata.value_validation import (
                validate_values,
            )
        except ImportError:
            raise ImportError(MISSING_TRACING_DEPS_ERROR_MSG) from None

        # Create a copy to avoid side effects
        metadata_df = dataframe.copy()
        total_spans = len(metadata_df)
        self._headers.update({"number-of-rows": str(total_spans)})
        metadata_df.reset_index(inplace=True, drop=True)

        if verbose:
            logger.info(
                f"Attempting to update metadata for {total_spans} spans"
            )

        # Check if we have any attributes.metadata.* columns to build a patch document
        metadata_prefix = "attributes.metadata."
        metadata_fields = [
            col
            for col in metadata_df.columns
            if col.startswith(metadata_prefix)
        ]
        has_metadata_fields = len(metadata_fields) > 0
        has_patch_document = patch_document_column_name in metadata_df.columns

        if not has_metadata_fields and not has_patch_document:
            error_msg = (
                f"No metadata fields found. Either provide columns with prefix '{metadata_prefix}' "
                f"or a '{patch_document_column_name}' column with JSON patch documents."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if has_metadata_fields and verbose:
            logger.info(
                f"Found {len(metadata_fields)} metadata field columns with prefix '{metadata_prefix}'"
            )

        # Create a new column for patch documents if we're going to use it
        if has_metadata_fields or has_patch_document:
            # Use 'patch_document' as the standardized column name for downstream processing
            final_patch_column = "patch_document"
            if final_patch_column not in metadata_df.columns:
                metadata_df[final_patch_column] = None

        # Process metadata field columns if they exist
        if has_metadata_fields:

            def build_patch_document(row):
                # Extract and preserve metadata values with proper types
                patch = {}
                for key in row.index:
                    if key.startswith("attributes.metadata."):
                        field_name = key.replace("attributes.metadata.", "")
                        # Check if the value is an array/list or other iterable (except strings)
                        if isinstance(row[key], (list, np.ndarray)) or (
                            hasattr(row[key], "__iter__")
                            and not isinstance(row[key], str)
                        ):
                            # For arrays/iterables, just add the value (nulls will be handled later)
                            patch[field_name] = row[key]
                        else:
                            # For scalar values, include even if it's None/null
                            # This is important for explicitly setting fields to null
                            patch[field_name] = row[key]
                return patch

            # Create patch documents from metadata fields
            field_patches = metadata_df.apply(build_patch_document, axis=1)

            # If there's an existing patch document column, we'll handle merging
            if has_patch_document:

                def process_patch_document(row_idx):
                    # Get the field patch for this row
                    field_patch = field_patches.iloc[row_idx]

                    # Get and process the explicit patch document
                    patch_doc = metadata_df.loc[
                        row_idx, patch_document_column_name
                    ]

                    # Handle different patch document formats
                    if patch_doc is None:
                        # None (as opposed to NaN) is a valid value but creates an empty patch
                        explicit_patch = {}
                    elif isinstance(patch_doc, float) and np.isnan(patch_doc):
                        # NaN is treated as an empty patch
                        explicit_patch = {}
                    elif isinstance(patch_doc, dict):
                        # Dict is used directly
                        explicit_patch = patch_doc
                    elif isinstance(patch_doc, str):
                        try:
                            explicit_patch = json.loads(patch_doc)
                            if not isinstance(explicit_patch, dict):
                                logger.warning(
                                    f"Row {row_idx}: Parsed patch document is not a dictionary. "
                                    f"Using empty dictionary instead."
                                )
                                explicit_patch = {}
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Row {row_idx}: Failed to parse patch document: {e}. "
                                f"Using empty dictionary instead."
                            )
                            explicit_patch = {}
                    else:
                        logger.warning(
                            f"Row {row_idx}: Unsupported patch document type: {type(patch_doc)}. "
                            f"Using empty dictionary instead."
                        )
                        explicit_patch = {}

                    # Merge patches - explicit patch takes precedence
                    merged_patch = {**field_patch, **explicit_patch}
                    return merged_patch

                # Apply the processing function to each row
                merged_patches = [
                    process_patch_document(i) for i in range(len(metadata_df))
                ]
                metadata_df[final_patch_column] = merged_patches
            else:
                # Just use the field patches directly
                metadata_df[final_patch_column] = field_patches
        elif (
            has_patch_document
            and patch_document_column_name != final_patch_column
        ):
            # If there are only patch documents (no metadata fields) and the column
            # isn't already named patch_document, rename it
            metadata_df[final_patch_column] = metadata_df[
                patch_document_column_name
            ]

        # Now process any patch documents that need to be parsed from strings to dicts
        if final_patch_column in metadata_df.columns:
            validation_errors = []

            def ensure_dict_patch(row_idx):
                patch = metadata_df.loc[row_idx, final_patch_column]

                # For None/null values, return an empty dict
                if patch is None:
                    return {}

                # Handle NaN differently from None
                if isinstance(patch, float) and np.isnan(patch):
                    return {}

                # If already a dict, return as is
                if isinstance(patch, dict):
                    return patch

                # If string, try to parse as JSON
                if isinstance(patch, str):
                    try:
                        parsed = json.loads(patch)
                        if isinstance(parsed, dict):
                            return parsed
                        else:
                            error_msg = (
                                f"Row {row_idx}: JSON must be an object/dictionary, "
                                f"got {type(parsed).__name__}"
                            )
                            logger.warning(error_msg)
                            validation_errors.append(error_msg)
                            return {} if not validate else None
                    except json.JSONDecodeError as e:
                        error_msg = f"Row {row_idx}: Invalid JSON in patch document: {e}"
                        logger.warning(error_msg)
                        validation_errors.append(error_msg)
                        return {} if not validate else None

                # For other types, log warning
                error_msg = f"Row {row_idx}: Unsupported patch type: {type(patch).__name__}"
                logger.warning(error_msg)
                validation_errors.append(error_msg)
                return {} if not validate else None

            # Process each row
            processed_patches = []
            for i in range(len(metadata_df)):
                patch = ensure_dict_patch(i)
                if patch is not None:
                    processed_patches.append(patch)

            # If validation is enabled and errors found, raise ValidationFailure
            if validate and validation_errors:
                raise err.ValidationFailure(validation_errors)

            metadata_df[final_patch_column] = processed_patches

        # Run validations on the processed dataframe
        if validate:
            if verbose:
                logger.info("Validating metadata update input")

            # Type validation
            errors = validate_argument_types(
                metadata_dataframe=metadata_df, project_name=project_name
            )
            if errors:
                for e in errors:
                    logger.error(f"Metadata validation error: {e}")
                raise err.ValidationFailure(errors)

            # Dataframe form validation
            if verbose:
                logger.info("Validating metadata update dataframe form")
            errors = validate_dataframe_form(
                metadata_dataframe=metadata_df,
                patch_document_column_name=final_patch_column,
            )
            if errors:
                for e in errors:
                    logger.error(f"Metadata dataframe validation error: {e}")
                raise err.ValidationFailure(errors)

            # Value validation
            if verbose:
                logger.info("Validating metadata update values")
            errors = validate_values(
                metadata_dataframe=metadata_df,
                patch_document_column_name=final_patch_column,
            )
            if errors:
                for e in errors:
                    logger.error(f"Metadata value validation error: {e}")
                raise err.ValidationFailure(errors)

        # Keep only the required columns
        metadata_df = self._remove_extraneous_columns(
            df=metadata_df,
            column_list=[SPAN_SPAN_ID_COL.name, final_patch_column],
        )

        if verbose:
            logger.info(
                "Using column names: context.span_id and patch_document"
            )

        # Ensure all patches are JSON strings for sending
        if final_patch_column in metadata_df.columns:
            metadata_df[final_patch_column] = metadata_df[
                final_patch_column
            ].apply(
                lambda p: json.dumps(p)
                if not isinstance(p, float) or not np.isnan(p)
                else json.dumps({})
            )

        # Convert to Arrow table
        try:
            if verbose:
                logger.info("Converting data to Arrow format")
            pa_schema = pa.Schema.from_pandas(metadata_df)
            ta = pa.Table.from_pandas(metadata_df)
        except pa.ArrowInvalid as e:
            logger.error(f"{INVALID_ARROW_CONVERSION_MSG}: {str(e)}")
            raise pa.ArrowInvalid(
                f"Error converting metadata to Arrow format: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error creating Arrow table: {str(e)}")
            raise

        # Create protocol schema
        proto_schema = proto._get_pb_schema_tracing(
            model_id=project_name,
            model_version=None,
        )

        # Use the _log_arrow_flight helper for consistency
        try:
            response = self._log_arrow_flight(
                pa_table=ta,
                pa_schema=pa_schema,
                proto_schema=proto_schema,
                verbose=verbose,
                model_id=project_name,
                model_version=None,
                request_type="metadata",
            )
        except Exception as e:
            logger.error(f"Error during metadata update request: {str(e)}")
            raise RuntimeError(
                f"Failed to submit metadata update: {str(e)}"
            ) from e

        if response:
            # Convert Protocol Buffer SpanError objects to dictionaries for easier access
            span_errors = [
                {"span_id": error.span_id, "error_message": error.error_message}
                for error in response.errors
            ]

            # Create result dictionary with direct field mapping from the response
            result = {
                "spans_processed": response.spans_processed,
                "spans_updated": response.spans_updated,
                "spans_failed": response.spans_failed,
                "errors": span_errors,
            }

            # Log summary of response
            if verbose:
                if response.spans_updated > 0 and response.spans_processed > 0:
                    msg = (
                        f"Metadata update: {response.spans_updated} of "
                        f"{response.spans_processed} spans updated "
                        f"({(response.spans_updated / response.spans_processed * 100):.1f}% success rate)"
                    )
                    logger.info(msg)
                else:
                    logger.warning(
                        f"Metadata update: {response.spans_updated} of "
                        f"{response.spans_processed} spans updated "
                        f"(0% success rate)"
                    )

                if span_errors:
                    logger.warning(
                        f"Encountered {len(span_errors)} errors during metadata update"
                    )
                    if verbose:
                        # Log all errors with their details
                        for error in span_errors:
                            logger.warning(
                                f"  Span {error['span_id']}: {error['error_message']}"
                            )

            return result
        else:
            # This should not happen with proper Flight client implementation,
            # but we handle it defensively
            logger.error(
                "No response received from server during metadata update"
            )
            raise RuntimeError(
                "Failed to receive response from metadata update operation"
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
        Logs a pandas dataframe containing inferences to Arize via a POST request. Use this to upload
        inferences from your model (ML, CV, NLP, etc.) to Arize.

        If you are looking to upload LLM traces or LLM evaluations, use :func:`log_spans` or
        :func:`log_evaluations`, respectively.

        Returns a :class:`Response` object from the Requests HTTP library to ensure successful delivery of
        records.

        Args:
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
            `Response` object

        """
        # This method requires the API key and either space ID or space key to be set
        # api_key and one of space_id or space_key must be provided
        if not self._api_key or not (self._space_id or self._space_key):
            raise AuthError(
                missing_space_id=not (self._space_id or self._space_key),
                missing_api_key=not self._api_key,
                method_name="log",
            )

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
        if (
            model_type == ModelTypes.GENERATIVE_LLM
            and environment != Environments.CORPUS
        ):
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
        has_cat_col = any(
            [ptypes.is_categorical_dtype(x) for x in dataframe.dtypes]
        )
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
                ) from None
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
            if (
                model_type == ModelTypes.GENERATIVE_LLM
                and environment != Environments.CORPUS
            ):
                dataframe, schema = self._add_generative_llm_columns(
                    dataframe, schema
                )

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
            proto_schema = proto._get_pb_schema_corpus(
                schema, model_id, model_type, environment
            )
        else:
            proto_schema = proto._get_pb_schema(
                schema,
                model_id,
                model_version,
                model_type,
                environment,
                batch_id,
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

    def _log_arrow_flight(
        self,
        pa_table: pa.Table,
        pa_schema: pa.Schema,
        proto_schema: pb2.Schema,
        verbose: bool = False,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        request_type: str = "evaluation",
    ) -> Union[
        WriteSpanEvaluationResponse,
        WriteSpanAnnotationResponse,
        WriteSpanAttributesMetadataResponse,
    ]:
        if verbose:
            logger.debug("Serializing schema.")

        base64_schema = base64.b64encode(proto_schema.SerializeToString())
        pa_schema = self._append_to_pyarrow_metadata(
            pa_schema, {"arize-schema": base64_schema}
        )

        if request_type == "evaluation":
            do_put_request = DoPutRequest(
                write_span_evaluation_request=WriteSpanEvaluationRequest(
                    space_id=self._space_id,
                    external_model_id=model_id,
                    model_version=model_version,
                )
            )
            response_type = WriteSpanEvaluationResponse
            log_context = "evaluation"
        elif request_type == "annotation":
            do_put_request = DoPutRequest(
                write_span_annotation_request=WriteSpanAnnotationRequest(
                    space_id=self._space_id,
                    external_model_id=model_id,
                    model_version=model_version,
                )
            )
            response_type = WriteSpanAnnotationResponse
            log_context = "annotation"
        elif request_type == "metadata":
            do_put_request = DoPutRequest(
                write_span_attributes_metadata_request=WriteSpanAttributesMetadataRequest(
                    space_id=self._space_id,
                    external_model_id=model_id,
                )
            )
            response_type = WriteSpanAttributesMetadataResponse
            log_context = "metadata"
        else:
            raise ValueError(
                f"Unsupported request_type in _log_arrow_flight: {request_type}"
            )

        encoded_command: bytes = json_format.MessageToJson(
            do_put_request
        ).encode("utf-8")
        descriptor = flight.FlightDescriptor.for_command(encoded_command)
        flight_client = self._flight_session.connect()
        res = None
        try:
            flight_writer, flight_metadata_reader = flight_client.do_put(
                descriptor, pa_schema, options=self._flight_session.call_options
            )
            with flight_writer:
                # write table as stream to flight server
                flight_writer.write_table(pa_table)
                # indicate that client has flushed all contents to stream
                flight_writer.done_writing()
                # read response from flight server
                flight_response = flight_metadata_reader.read()
                if flight_response is not None:
                    # Use the correct response type
                    res = response_type()
                    res.ParseFromString(flight_response.to_pybytes())
                    logger.info(
                        f"Successfully logged {log_context} data to Arize for model '{model_id}' "
                    )
        except Exception:
            logger.exception(f"Error logging {log_context} data to Arize")
        finally:
            flight_client.close()
        return res

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
        pa_schema = self._append_to_pyarrow_metadata(
            pa_schema, {"arize-schema": base64_schema}
        )

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
            writer.write_table(pa_table, max_chunksize=16384)
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
        df.insert(
            loc=0,
            column=GENERATED_PREDICTION_LABEL_COL,
            value=1,
            allow_duplicates=False,
        )
        return df

    # Add in all the relevant columns for generative LLMs and modify the schema accordingly
    def _add_generative_llm_columns(
        self, dataframe: pd.DataFrame, schema: Schema
    ) -> (pd.DataFrame, Schema):
        if (
            schema.prediction_label_column_name is None
            and schema.actual_label_column_name is None
        ):
            dataframe = self._add_default_prediction_label_column(dataframe)
            schema = schema.replace(
                prediction_label_column_name=GENERATED_PREDICTION_LABEL_COL
            )
        if (
            schema.llm_config_column_names
            and schema.llm_config_column_names.params_column_name
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
            if (
                schema.llm_run_metadata_column_names.total_token_count_column_name
                is not None
            ):
                dataframe = self._add_reserved_tag_column(
                    dataframe,
                    LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
                    schema.llm_run_metadata_column_names.total_token_count_column_name,
                )
                schema.tag_column_names.append(
                    LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME
                )
            if (
                schema.llm_run_metadata_column_names.prompt_token_count_column_name
                is not None
            ):
                dataframe = self._add_reserved_tag_column(
                    dataframe,
                    LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
                    schema.llm_run_metadata_column_names.prompt_token_count_column_name,
                )
                schema.tag_column_names.append(
                    LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME
                )
            if (
                schema.llm_run_metadata_column_names.response_token_count_column_name
                is not None
            ):
                dataframe = self._add_reserved_tag_column(
                    dataframe,
                    LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
                    schema.llm_run_metadata_column_names.response_token_count_column_name,
                )
                schema.tag_column_names.append(
                    LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME
                )
            if (
                schema.llm_run_metadata_column_names.response_latency_ms_column_name
                is not None
            ):
                dataframe = self._add_reserved_tag_column(
                    dataframe,
                    LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
                    schema.llm_run_metadata_column_names.response_latency_ms_column_name,
                )
                schema.tag_column_names.append(
                    LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME
                )
        return dataframe, schema

    @staticmethod
    def _add_json_llm_params_column(
        df: pd.DataFrame, llm_params_col_name: str
    ) -> pd.DataFrame:
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
            matched_regex_cols = []
            for col in df.columns:
                match_result = re.match(regex, col)
                if match_result:
                    matched_regex_cols.append(col)
            relevant_columns.update(matched_regex_cols)

        final_columns = list(set(df.columns) & relevant_columns)
        return df[final_columns]

    @staticmethod
    def _generative_model_warnings(df: pd.DataFrame, schema: Schema):
        # Warning for when prediction_label_column_name is not provided
        if (
            schema.prediction_label_column_name is None
            and schema.actual_label_column_name is None
        ):
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
    def _append_to_pyarrow_metadata(
        pa_schema: pa.Schema, new_metadata: Dict[str, Any]
    ):
        # Ensure metadata is handled correctly, even if initially None.
        metadata = pa_schema.metadata
        if metadata is None:
            # Initialize an empty dict if schema metadata was None
            metadata = {}

        conflicting_keys = metadata.keys() & new_metadata.keys()
        if conflicting_keys:
            raise KeyError(
                "Cannot append metadata to pyarrow schema. "
                f"There are conflicting keys: {log_a_list(conflicting_keys, join_word='and')}"
            )

        updated_metadata = metadata.copy()
        updated_metadata.update(new_metadata)
        return pa_schema.with_metadata(updated_metadata)


@dataclass
class FlightSession:
    api_key: str
    space_id: str
    host: str
    port: int
    scheme: str
    session_name: str = field(init=False)
    call_options: flight.FlightCallOptions = field(init=False)

    def __post_init__(self):
        self.session_name = f"python-sdk-{uuid.uuid4()}"
        if self.api_key is None:
            logger.error(InvalidSessionError.error_message())
            raise InvalidSessionError

        if self.space_id is None:
            logger.error(InvalidSessionError.error_message())
            raise InvalidSessionError
        self._headers = [
            (b"origin", b"arize-logging-client"),
            (b"auth-token-bin", f"{self.api_key}".encode()),
            (b"space-id", f"{self.space_id}".encode()),
            (b"sdk-language", b"python"),
            (b"language-version", get_python_version().encode("utf-8")),
            (b"sdk-version", __version__.encode("utf-8")),
        ]

    def connect(self) -> flight.FlightClient:
        """
        Connects to public ingestion endpoint
        """
        try:
            # disable TLS verification for local development
            disable_cert = self.host.lower() == "localhost"
            self.call_options = flight.FlightCallOptions(headers=self._headers)
            return flight.FlightClient(
                location=f"{self.scheme}://{self.host}:{self.port}",
                disable_server_verification=disable_cert,
            )
        except Exception as e:
            logger.error(
                f"There was an error trying to connect to the Arize ingestion endpoint, {e}"
            )
            raise


class InvalidSessionError(ValueError):
    @staticmethod
    def error_message() -> str:
        return (
            "API key isn't provided or is invalid. "
            "Please pass in the correct API key from the UI when "
            "initiating a new Arize python client. Alternatively, you can set up credentials "
            "through a profile or an environment variable"
        )
