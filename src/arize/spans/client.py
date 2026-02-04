"""Client implementation for managing spans and traces in the Arize platform."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
from google.protobuf import json_format, message

from arize._exporter.client import ArizeExportClient
from arize._flight.client import ArizeFlightClient, FlightPostArrowFileResponse
from arize._flight.types import FlightRequestType
from arize.constants.spans import DEFAULT_DATETIME_FMT
from arize.exceptions.base import (
    INVALID_ARROW_CONVERSION_MSG,
    ValidationError,
    ValidationFailure,
)
from arize.exceptions.models import MissingProjectNameError
from arize.exceptions.spaces import MissingSpaceIDError
from arize.logging import CtxAdapter
from arize.ml.types import Environments
from arize.spans.validation.metadata.value_validation import (
    InvalidPatchDocumentFormat,
)
from arize.utils.arrow import post_arrow_table
from arize.utils.dataframe import (
    remove_extraneous_columns,
    reset_dataframe_index,
)
from arize.utils.proto import get_pb_schema_tracing

if TYPE_CHECKING:
    import requests

    from arize._generated.protocol.flight import flight_pb2
    from arize.config import SDKConfiguration

logger = logging.getLogger(__name__)


class SpansClient:
    """Client for logging LLM tracing spans and evaluations to Arize.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.
    """

    def __init__(self, *, sdk_config: SDKConfiguration) -> None:
        """
        Args:
            sdk_config: Resolved SDK configuration.
        """  # noqa: D205, D212
        self._sdk_config = sdk_config

    def log(
        self,
        *,
        space_id: str,
        project_name: str,
        dataframe: pd.DataFrame,
        evals_dataframe: pd.DataFrame | None = None,
        datetime_format: str = DEFAULT_DATETIME_FMT,
        validate: bool = True,
        timeout: float | None = None,
        tmp_dir: str = "",
    ) -> requests.Response:
        """Logs a pandas dataframe containing LLM tracing data to Arize via a POST request.

        Returns a :class:`Response` object from the Requests HTTP library to ensure
        successful delivery of records.

        Args:
            space_id: The space ID where the project resides.
            project_name: A unique name to identify your project in the Arize platform.
            dataframe (:class:`pandas.DataFrame`): The dataframe containing the LLM traces.
            evals_dataframe (:class:`pandas.DataFrame` | :obj:`None`): A dataframe containing
                LLM evaluations data. The evaluations are joined to their corresponding spans
                via a left outer join, i.e., using only `context.span_id` from the spans
                dataframe. Defaults to None.
            datetime_format: format for the timestamp captured in the LLM traces.
                Defaults to "%Y-%m-%dT%H:%M:%S.%f+00:00".
            validate: When set to True, validation is run before sending data.
                Defaults to True.
            timeout: You can stop waiting for a response after a given number
                of seconds with the timeout parameter. Defaults to None.
            tmp_dir: Temporary directory/file to store the serialized data in binary
                before sending to Arize.

        Returns:
            Response object from the HTTP request.

        """
        from arize.spans.columns import (
            EVAL_COLUMN_PATTERN,
            ROOT_LEVEL_SPAN_KIND_COL,
            SPAN_KIND_COL,
            SPAN_OPENINFERENCE_COLUMNS,
            SPAN_SPAN_ID_COL,
        )
        from arize.spans.conversion import (
            convert_timestamps,
            jsonify_dictionaries,
        )
        from arize.spans.validation.evals import evals_validation
        from arize.spans.validation.spans import spans_validation

        # This method requires a space_id and project_name
        if not space_id:
            raise MissingSpaceIDError()
        if not project_name:
            raise MissingProjectNameError()

        # We need our own copy since we will manipulate the underlying data and
        # do not want side effects
        spans_df = dataframe.copy()
        evals_df = (
            evals_dataframe.copy() if evals_dataframe is not None else None
        )

        # Bind common context for this operation
        log = CtxAdapter(
            logger,
            {
                "resource": "spans",
                "operation": "log",
                "space_id": space_id,
                "project": project_name,
                "validate": validate,
                "spans_df_rows": len(spans_df),
                "evals_df_rows": len(evals_df) if evals_df is not None else 0,
            },
        )

        # We expect the index to be 0,1,2,3..., len(df)-1. Phoenix, for example, will give us a dataframe
        # with context_id as the index
        reset_dataframe_index(dataframe=spans_df)
        if evals_df:
            reset_dataframe_index(dataframe=evals_df)

        log.debug("Performing direct input type validation")
        errors = spans_validation.validate_argument_types(
            spans_dataframe=spans_df,
            project_name=project_name,
            dt_fmt=datetime_format,
        )
        if evals_df is not None:
            eval_errors = evals_validation.validate_argument_types(
                evals_dataframe=evals_df,
                project_name=project_name,
            )
            errors += eval_errors
        for e in errors:
            log.error(e)
        if errors:
            raise ValidationFailure(errors)

        if validate:
            log.debug("Performing dataframe form validation")
            errors = spans_validation.validate_dataframe_form(
                spans_dataframe=spans_df
            )
            if evals_df is not None:
                eval_errors = evals_validation.validate_dataframe_form(
                    evals_dataframe=evals_df
                )
                errors += eval_errors
            for e in errors:
                log.error(e)
            if errors:
                raise ValidationFailure(errors)

        log.debug("Removing unnecessary columns")
        spans_df = remove_extraneous_columns(
            df=spans_df,
            column_list=[col.name for col in SPAN_OPENINFERENCE_COLUMNS],
        )
        if evals_df:
            evals_df = remove_extraneous_columns(
                df=evals_df,
                column_list=[SPAN_SPAN_ID_COL.name],
                regex=EVAL_COLUMN_PATTERN,
            )

        log.debug("Converting timestamps")
        spans_df = convert_timestamps(df=spans_df, fmt=datetime_format)

        if validate:
            log.debug("Performing values validation")
            errors = spans_validation.validate_values(
                spans_dataframe=spans_df,
                project_name=project_name,
            )
            if evals_df:
                eval_errors = evals_validation.validate_values(
                    evals_dataframe=evals_df,
                    project_name=project_name,
                )
                errors += eval_errors
            for e in errors:
                log.error(e)
            if errors:
                raise ValidationFailure(errors)

        log.debug("Converting dictionaries to JSON objects")
        spans_df = jsonify_dictionaries(spans_df)
        if (
            ROOT_LEVEL_SPAN_KIND_COL.name in spans_df.columns
            and SPAN_KIND_COL.name not in spans_df.columns
        ):
            log.debug("Moving span kind to atributes")
            spans_df.rename(
                columns={ROOT_LEVEL_SPAN_KIND_COL.name: SPAN_KIND_COL.name},
                inplace=True,
            )

        df = (
            pd.merge(spans_df, evals_df, on=SPAN_SPAN_ID_COL.name, how="left")
            if evals_df
            else spans_df
        )

        # Convert to Arrow table
        try:
            log.debug("Converting data to Arrow format")
            pa_table = pa.Table.from_pandas(df, preserve_index=False)
        except pa.ArrowInvalid as e:
            log.exception(INVALID_ARROW_CONVERSION_MSG)
            raise pa.ArrowInvalid(
                f"Error converting to Arrow format: {e!s}"
            ) from e
        except Exception:
            log.exception("Unexpected error creating Arrow table")
            raise

        proto_schema = get_pb_schema_tracing(project_name=project_name)
        # Create headers copy for the spans client
        # Safe to mutate, returns a deep copy
        headers = self._sdk_config.headers
        # Send the number of rows in the dataframe as a header
        # This helps the Arize server to return appropriate feedback, specially for async logging
        headers.update(
            {
                "arize-space-id": space_id,
                "arize-interface": "batch",
                "number-of-rows": str(len(spans_df)),
            }
        )
        return post_arrow_table(
            files_url=self._sdk_config.files_url,
            pa_table=pa_table,
            proto_schema=proto_schema,
            headers=headers,
            timeout=timeout,
            verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
            tmp_dir=tmp_dir,
        )

    def update_evaluations(
        self,
        *,
        space_id: str,
        project_name: str,
        dataframe: pd.DataFrame,
        validate: bool = True,
        force_http: bool = False,
        timeout: float | None = None,
        tmp_dir: str = "",
    ) -> flight_pb2.WriteSpanEvaluationResponse:
        """Logs a pandas dataframe containing LLM evaluations data to Arize via a Flight gRPC request.

        The dataframe must contain a column `context.span_id` such that Arize can assign
        each evaluation to its respective span.

        Args:
            space_id: The space ID where the project resides.
            project_name: A unique name to identify your project in the Arize platform.
            dataframe (:class:`pandas.DataFrame`): A dataframe containing LLM evaluations data.
            validate: When set to True, validation is run before sending data.
                Defaults to True.
            force_http: Force the use of HTTP for data upload. Defaults to False.
            timeout: You can stop waiting for a response after a given number
                of seconds with the timeout parameter. Defaults to None.
            tmp_dir: Temporary directory/file to store the serialized data in binary
                before sending to Arize.
        """
        from arize.spans.columns import EVAL_COLUMN_PATTERN, SPAN_SPAN_ID_COL
        from arize.spans.validation.evals import evals_validation

        # This method requires a space_id and project_name
        if not space_id:
            raise MissingSpaceIDError()
        if not project_name:
            raise MissingProjectNameError()

        # Bind common context for this operation
        log = CtxAdapter(
            logger,
            {
                "resource": "spans",
                "operation": "log",
                "space_id": space_id,
                "project": project_name,
                "validate": validate,
                "evals_df_rows": len(dataframe),
            },
        )

        # We need our own copy since we will manipulate the underlying data and
        # do not want side effects
        evals_df = dataframe.copy()

        # We expect the index to be 0,1,2,3..., len(df)-1. Phoenix, for example, will give us a dataframe
        # with context_id as the index; the old index is not meaningful in our copy of the original dataframe
        # so we can drop it.
        reset_dataframe_index(dataframe=evals_df)

        log.debug("Performing direct input type validation")
        errors = evals_validation.validate_argument_types(
            evals_dataframe=evals_df,
            project_name=project_name,
        )
        for e in errors:
            log.error(e)
        if errors:
            raise ValidationFailure(errors)

        if validate:
            log.debug("Performing dataframe form validation")
            errors = evals_validation.validate_dataframe_form(
                evals_dataframe=evals_df
            )
            for e in errors:
                log.error(e)
            if errors:
                raise ValidationFailure(errors)

        log.debug("Removing unnecessary columns")
        evals_df = remove_extraneous_columns(
            df=evals_df,
            column_list=[SPAN_SPAN_ID_COL.name],
            regex=EVAL_COLUMN_PATTERN,
        )

        if validate:
            log.debug("Performing values validation")
            errors = evals_validation.validate_values(
                evals_dataframe=evals_df,
                project_name=project_name,
            )
            for e in errors:
                log.error(e)
            if errors:
                raise ValidationFailure(errors)

        # Convert to Arrow table
        try:
            log.debug("Converting data to Arrow format")
            pa_table = pa.Table.from_pandas(evals_df, preserve_index=False)
        except pa.ArrowInvalid as e:
            log.exception(INVALID_ARROW_CONVERSION_MSG)
            raise pa.ArrowInvalid(
                f"Error converting to Arrow format: {e!s}"
            ) from e
        except Exception:
            log.exception("Unexpected error creating Arrow table")
            raise

        if force_http:
            proto_schema = get_pb_schema_tracing(project_name=project_name)
            # Create headers copy for the spans client
            # Safe to mutate, returns a deep copy
            headers = self._sdk_config.headers
            # Send the number of rows in the dataframe as a header
            # This helps the Arize server to return appropriate feedback, specially for async logging
            headers.update(
                {
                    "arize-space-id": space_id,
                    "arize-interface": "batch",
                    "number-of-rows": str(len(dataframe)),
                }
            )
            return post_arrow_table(
                files_url=self._sdk_config.files_url,
                pa_table=pa_table,
                proto_schema=proto_schema,
                headers=headers,
                timeout=timeout,
                verify=self._sdk_config.request_verify,
                max_chunksize=self._sdk_config.pyarrow_max_chunksize,
                tmp_dir=tmp_dir,
            )

        request_type = FlightRequestType.EVALUATION
        response = None
        with ArizeFlightClient(
            api_key=self._sdk_config.api_key,
            host=self._sdk_config.flight_host,
            port=self._sdk_config.flight_port,
            scheme=self._sdk_config.flight_scheme,
            request_verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
        ) as flight_client:
            try:
                response = flight_client.log_arrow_table(
                    space_id=space_id,
                    project_name=project_name,
                    pa_table=pa_table,
                    request_type=request_type,
                )
            except Exception as e:
                msg = f"Error during update request: {e!s}"
                log.exception(msg)
                raise RuntimeError(msg) from e

        if response is None:
            # This should not happen with proper Flight client implementation,
            # but we handle it defensively
            msg = "No response received from flight server during update"
            log.error(msg)
            raise RuntimeError(msg)

        _log_flight_update_summary(
            project_name=project_name,
            total_spans=len(pa_table),
            request_type=request_type,
            response=response,
        )

        # Convert Protocol Buffer SpanError objects to dictionaries for easier access
        return _message_to_dict(response)

    def update_annotations(
        self,
        *,
        space_id: str,
        project_name: str,
        dataframe: pd.DataFrame,
        validate: bool = True,
    ) -> flight_pb2.WriteSpanAnnotationResponse:
        """Logs a pandas dataframe containing LLM span annotations to Arize via a Flight gRPC request.

        The dataframe must contain a column `context.span_id` such that Arize can assign
        each annotation to its respective span. Annotation columns should follow the pattern
        `annotation.<name>.<suffix>` where suffix is either `label` or `score`. An optional
        `annotation.notes` column can be included for free-form text notes.

        Args:
            space_id: The space ID where the project resides.
            project_name: A unique name to identify your project in the Arize platform.
            dataframe (:class:`pandas.DataFrame`): A dataframe containing LLM annotation data.
            validate: When set to True, validation is run before sending data.
                Defaults to True.
        """
        from arize.spans.columns import (
            ANNOTATION_COLUMN_PATTERN,
            ANNOTATION_LABEL_SUFFIX,
            ANNOTATION_NOTES_COLUMN_NAME,
            ANNOTATION_SCORE_SUFFIX,
            ANNOTATION_UPDATED_AT_SUFFIX,
            ANNOTATION_UPDATED_BY_SUFFIX,
            SPAN_SPAN_ID_COL,
        )
        from arize.spans.validation.annotations import annotations_validation

        # This method requires a space_id and project_name
        if not space_id:
            raise MissingSpaceIDError()
        if not project_name:
            raise MissingProjectNameError()

        # Bind common context for this operation
        log = CtxAdapter(
            logger,
            {
                "resource": "spans",
                "operation": "log",
                "space_id": space_id,
                "project": project_name,
                "validate": validate,
                "evals_df_rows": len(dataframe),
            },
        )

        anno_df = dataframe.copy()

        # We expect the index to be 0,1,2,3..., len(df)-1. Phoenix, for example, will give us a dataframe
        # with context_id as the index; the old index is not meaningful in our copy of the original dataframe
        # so we can drop it.
        reset_dataframe_index(dataframe=anno_df)

        log.debug(
            "Checking for and autogenerating missing updated_by/updated_at annotation columns"
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

        log.debug(f"Found annotation names: {annotation_names}")

        current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        for name in annotation_names:
            updated_by_col = f"annotation.{name}{ANNOTATION_UPDATED_BY_SUFFIX}"
            updated_at_col = f"annotation.{name}{ANNOTATION_UPDATED_AT_SUFFIX}"
            label_col = f"annotation.{name}{ANNOTATION_LABEL_SUFFIX}"
            score_col = f"annotation.{name}{ANNOTATION_SCORE_SUFFIX}"

            # Check if *any* part of this annotation exists (label or score)
            # Only add metadata if the annotation itself is present
            if label_col in anno_df.columns or score_col in anno_df.columns:
                if updated_by_col not in anno_df.columns:
                    log.debug(f"Autogenerating column: {updated_by_col}")
                    anno_df[updated_by_col] = "SDK"
                if updated_at_col not in anno_df.columns:
                    log.debug(f"Autogenerating column: {updated_at_col}")
                    anno_df[updated_at_col] = current_time_ms
            else:
                log.debug(
                    f"Skipping metadata generation for '{name}' as no label or score column found."
                )

        if ANNOTATION_NOTES_COLUMN_NAME in anno_df.columns:
            log.debug(
                f"Formatting {ANNOTATION_NOTES_COLUMN_NAME} column to JSON strings within lists."
            )
            anno_df[ANNOTATION_NOTES_COLUMN_NAME] = anno_df[
                ANNOTATION_NOTES_COLUMN_NAME
            ].apply(
                partial(
                    _format_note_for_storage,
                    current_time_ms=current_time_ms,
                )
            )

        log.debug("Performing direct input type validation for annotations")
        errors = annotations_validation.validate_argument_types(
            annotations_dataframe=anno_df,
            project_name=project_name,
        )
        for e in errors:
            log.error(e)
        if errors:
            raise ValidationFailure(errors)

        if validate:
            log.debug("Performing dataframe form validation for annotations")
            errors = annotations_validation.validate_dataframe_form(
                annotations_dataframe=anno_df
            )
            for e in errors:
                log.error(e)
            if errors:
                raise ValidationFailure(errors)

        log.debug("Removing unnecessary annotation columns")
        # Update columns to keep: span_id, annotation.notes, and annotation pattern
        columns_to_keep = [SPAN_SPAN_ID_COL.name]
        if ANNOTATION_NOTES_COLUMN_NAME in anno_df.columns:
            columns_to_keep.append(ANNOTATION_NOTES_COLUMN_NAME)
        anno_df = remove_extraneous_columns(
            df=anno_df,
            column_list=columns_to_keep,
            regex=ANNOTATION_COLUMN_PATTERN,
        )

        if validate:
            log.debug("Performing annotation values validation")
            errors = annotations_validation.validate_values(
                annotations_dataframe=anno_df,
                project_name=project_name,
            )
            for e in errors:
                log.error(e)
            if errors:
                raise ValidationFailure(errors)

        # Convert to Arrow table
        try:
            log.debug("Converting data to Arrow format")
            pa_table = pa.Table.from_pandas(anno_df, preserve_index=False)
        except pa.ArrowInvalid as e:
            log.exception(INVALID_ARROW_CONVERSION_MSG)
            raise pa.ArrowInvalid(
                f"Error converting to Arrow format: {e!s}"
            ) from e
        except Exception:
            log.exception("Unexpected error creating Arrow table")
            raise

        if ANNOTATION_NOTES_COLUMN_NAME in anno_df.columns:
            notes_field = pa_table.schema.field(ANNOTATION_NOTES_COLUMN_NAME)
            if not (
                isinstance(notes_field.type, pa.ListType)
                and notes_field.type.value_type == pa.string()
            ):
                log.warning(
                    f"Warning: Inferred type for {ANNOTATION_NOTES_COLUMN_NAME} is "
                    f"{notes_field.type}, expected list<string>."
                )

        request_type = FlightRequestType.ANNOTATION
        response = None
        with ArizeFlightClient(
            api_key=self._sdk_config.api_key,
            host=self._sdk_config.flight_host,
            port=self._sdk_config.flight_port,
            scheme=self._sdk_config.flight_scheme,
            request_verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
        ) as flight_client:
            try:
                response = flight_client.log_arrow_table(
                    space_id=space_id,
                    project_name=project_name,
                    pa_table=pa_table,
                    request_type=request_type,
                )
            except Exception as e:
                msg = f"Error during update request: {e!s}"
                log.exception(msg)
                raise RuntimeError(msg) from e

        if response is None:
            # This should not happen with proper Flight client implementation,
            # but we handle it defensively
            msg = "No response received from flight server during update"
            log.error(msg)
            raise RuntimeError(msg)

        _log_flight_update_summary(
            project_name=project_name,
            total_spans=len(pa_table),
            request_type=request_type,
            response=response,
        )

        # Convert Protocol Buffer SpanError objects to dictionaries for easier access
        return _message_to_dict(response)

    def update_metadata(
        self,
        *,
        space_id: str,
        project_name: str,
        dataframe: pd.DataFrame,
        patch_document_column_name: str = "patch_document",
        validate: bool = True,
    ) -> dict[str, Any]:
        """Log metadata updates using JSON Merge Patch format.

        This method is only supported for LLM model types.

        The dataframe must contain a column `context.span_id` to identify spans and either:

        1. A column with JSON patch documents (specified by patch_document_column_name), or
        2. One or more columns with prefix `attributes.metadata.` that will be automatically
           converted to a patch document (e.g., `attributes.metadata.tag` → `{"tag": value}`).

        If both methods are used, the explicit patch document is applied after the individual field updates.
        The patches will be applied to the `attributes.metadata` field of each span.

        Type Handling:

        - The client primarily supports string, integer, and float data types.
        - Boolean values are converted to string representations.
        - Nested JSON objects and arrays are serialized to JSON strings during transmission.
        - Setting a field to `None` or `null` will set the field to JSON null in the metadata.
          Note: This differs from standard JSON Merge Patch where null values remove fields.

        Args:
            space_id: The space ID where the project resides.
            project_name: A unique name to identify your project in the Arize platform.
            dataframe (:class:`pandas.DataFrame`): DataFrame with span_ids and either patch
                documents or metadata field columns.
            patch_document_column_name: Name of the column containing JSON patch documents.
                Defaults to "patch_document".
            validate: When set to True, validation is run before sending data.

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
            AuthError: When API key or space ID is missing.
            ValidationFailure: When validation of the dataframe or values fails.
            ImportError: When required tracing dependencies are missing.
            ArrowInvalid: When the dataframe cannot be converted to Arrow format.
            RuntimeError: If the request fails or no response is received.

        Examples:
            Method 1: Using a patch document

            >>> df = pd.DataFrame(
            ...     {
            ...         "context.span_id": ["span1", "span2"],
            ...         "patch_document": [
            ...             {"tag": "important"},
            ...             {"priority": "high"},
            ...         ],
            ...     }
            ... )

            Method 2: Using direct field columns

            >>> df = pd.DataFrame(
            ...     {
            ...         "context.span_id": ["span1", "span2"],
            ...         "attributes.metadata.tag": ["important", "standard"],
            ...         "attributes.metadata.priority": ["high", "medium"],
            ...     }
            ... )

            Method 3: Combining both approaches

            >>> df = pd.DataFrame(
            ...     {
            ...         "context.span_id": ["span1"],
            ...         "attributes.metadata.tag": ["important"],
            ...         "patch_document": [
            ...             {"priority": "high"}
            ...         ],  # Overrides conflicting fields
            ...     }
            ... )

            Method 4: Setting fields to null

            >>> df = pd.DataFrame(
            ...     {
            ...         "context.span_id": ["span1"],
            ...         "attributes.metadata.old_field": [
            ...             None
            ...         ],  # Sets field to JSON null
            ...         "patch_document": [
            ...             {"other_field": None}
            ...         ],  # Also sets field to JSON null
            ...     }
            ... )
        """
        # Import validation modules
        from arize.spans.columns import SPAN_SPAN_ID_COL
        from arize.spans.validation.metadata.argument_validation import (
            validate_argument_types,
        )
        from arize.spans.validation.metadata.dataframe_form_validation import (
            validate_dataframe_form,
        )
        from arize.spans.validation.metadata.value_validation import (
            validate_values,
        )

        # This method requires a space_id and project_name
        if not space_id:
            raise MissingSpaceIDError()
        if not project_name:
            raise MissingProjectNameError()

        # Bind common context for this operation
        log = CtxAdapter(
            logger,
            {
                "resource": "spans",
                "operation": "log",
                "space_id": space_id,
                "project": project_name,
                "validate": validate,
                "evals_df_rows": len(dataframe),
            },
        )

        # We need our own copy since we will manipulate the underlying data and
        # do not want side effects
        metadata_df = dataframe.copy()

        # We expect the index to be 0,1,2,3..., len(df)-1. Phoenix, for example, will give us a dataframe
        # with context_id as the index; the old index is not meaningful in our copy of the original dataframe
        # so we can drop it.
        reset_dataframe_index(dataframe=metadata_df)

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
            log.error(error_msg)
            raise ValueError(error_msg)

        if has_metadata_fields:
            log.debug(
                f"Found {len(metadata_fields)} metadata field columns with prefix '{metadata_prefix}'"
            )

        # Create a new column for patch documents if we're going to use it
        # Use 'patch_document' as the standardized column name for downstream processing
        final_patch_column = "patch_document"
        if final_patch_column not in metadata_df.columns:
            metadata_df[final_patch_column] = None

        # Process metadata field columns if they exist
        if has_metadata_fields:
            # Create patch documents from metadata fields
            field_patches = metadata_df.apply(_build_patch_document, axis=1)

            # If there's an existing patch document column, we'll handle merging
            if has_patch_document:
                # Apply the processing function to each row
                merged_patches = [
                    _process_patch_document(
                        metadata_df,
                        patch_document_column_name,
                        field_patches,
                        idx,
                    )
                    for idx in range(len(metadata_df))
                ]
                # Type ignore: pandas DataFrame column assignment type is overly restrictive
                metadata_df[final_patch_column] = merged_patches  # type: ignore[assignment]
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

            # Process each row
            processed_patches = []
            for idx in range(len(metadata_df)):
                patch, errors = _ensure_dict_patch(
                    metadata_df,
                    final_patch_column,
                    idx,
                )
                if patch:
                    processed_patches.append(patch)
                if errors:
                    validation_errors.extend(errors)

            # If validation is enabled and errors found, raise ValidationFailure
            if validate and validation_errors:
                for e in validation_errors:
                    log.error(e)
                raise ValidationFailure(validation_errors)

            # Type ignore: pandas DataFrame column assignment type is overly restrictive
            metadata_df[final_patch_column] = processed_patches  # type: ignore[assignment]

        # Run validations on the processed dataframe
        if validate:
            log.debug("Validating metadata update input")

            # Type validation
            errors = validate_argument_types(
                metadata_dataframe=metadata_df, project_name=project_name
            )
            for e in errors:
                log.error(e)
            if errors:
                raise ValidationFailure(errors)

            # Dataframe form validation
            log.debug("Validating metadata update dataframe form")
            errors = validate_dataframe_form(
                metadata_dataframe=metadata_df,
                patch_document_column_name=final_patch_column,
            )
            for e in errors:
                log.error(e)
            if errors:
                raise ValidationFailure(errors)

            # Value validation
            log.debug("Validating metadata update values")
            errors = validate_values(
                metadata_dataframe=metadata_df,
                patch_document_column_name=final_patch_column,
            )
            for e in errors:
                log.error(e)
            if errors:
                raise ValidationFailure(errors)

        # Keep only the required columns
        metadata_df = remove_extraneous_columns(
            df=metadata_df,
            column_list=[SPAN_SPAN_ID_COL.name, final_patch_column],
        )

        log.debug("Using column names: context.span_id and patch_document")
        # Ensure all patches are JSON strings for sending
        if final_patch_column in metadata_df.columns:
            metadata_df[final_patch_column] = metadata_df[
                final_patch_column
            ].apply(
                lambda p: (
                    json.dumps(p)
                    if not isinstance(p, float) or not np.isnan(p)
                    else json.dumps({})
                )
            )

        # Convert to Arrow table
        try:
            log.debug("Converting data to Arrow format")
            pa_table = pa.Table.from_pandas(metadata_df, preserve_index=False)
        except pa.ArrowInvalid as e:
            log.exception(INVALID_ARROW_CONVERSION_MSG)
            raise pa.ArrowInvalid(
                f"Error converting to Arrow format: {e!s}"
            ) from e
        except Exception:
            log.exception("Unexpected error creating Arrow table")
            raise

        request_type = FlightRequestType.METADATA
        response = None
        with ArizeFlightClient(
            api_key=self._sdk_config.api_key,
            host=self._sdk_config.flight_host,
            port=self._sdk_config.flight_port,
            scheme=self._sdk_config.flight_scheme,
            request_verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
        ) as flight_client:
            try:
                response = flight_client.log_arrow_table(
                    space_id=space_id,
                    project_name=project_name,
                    pa_table=pa_table,
                    request_type=request_type,
                )
            except Exception as e:
                msg = f"Error during update request: {e!s}"
                log.exception(msg)
                raise RuntimeError(msg) from e

        if response is None:
            # This should not happen with proper Flight client implementation,
            # but we handle it defensively
            msg = "No response received from flight server during update"
            log.error(msg)
            raise RuntimeError(msg)

        _log_flight_update_summary(
            project_name=project_name,
            total_spans=len(pa_table),
            request_type=request_type,
            response=response,
        )

        # Convert Protocol Buffer SpanError objects to dictionaries for easier access
        return _message_to_dict(response)

    def export_to_df(
        self,
        *,
        space_id: str,
        project_name: str,
        start_time: datetime,
        end_time: datetime,
        where: str = "",
        columns: list | None = None,
        stream_chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """Export span data from Arize to a :class:`pandas.DataFrame`.

        Retrieves trace/span data from the specified project within a time range
        and returns it as a :class:`pandas.DataFrame`. Supports filtering with SQL-like
        WHERE clauses and similarity search for semantic retrieval.

        Returns:
            :class:`pandas.DataFrame`: DataFrame containing the requested span data with columns
                for span metadata, attributes, events, and any custom fields.
        """
        with ArizeFlightClient(
            api_key=self._sdk_config.api_key,
            host=self._sdk_config.flight_host,
            port=self._sdk_config.flight_port,
            scheme=self._sdk_config.flight_scheme,
            request_verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
        ) as flight_client:
            exporter = ArizeExportClient(
                flight_client=flight_client,
            )
            return exporter.export_to_df(
                space_id=space_id,
                model_id=project_name,
                environment=Environments.TRACING,
                start_time=start_time,
                end_time=end_time,
                where=where,
                columns=columns,
                stream_chunk_size=stream_chunk_size,
            )

    def export_to_parquet(
        self,
        *,
        path: str,
        space_id: str,
        project_name: str,
        start_time: datetime,
        end_time: datetime,
        where: str = "",
        columns: list | None = None,
        stream_chunk_size: int | None = None,
    ) -> None:
        """Export span data from Arize to a Parquet file.

        Retrieves trace/span data from the specified project within a time range
        and writes it directly to a Parquet file at the specified path. Supports
        filtering with SQL-like WHERE clauses for efficient querying. Ideal for
        large datasets and long-term storage.

        Args:
            path: The file path where the Parquet file will be written.
            space_id: The space ID where the project resides.
            project_name: The name of the project to export span data from.
            start_time: Start of the time range (inclusive) as a datetime object.
            end_time: End of the time range (inclusive) as a datetime object.
            where: Optional SQL-like WHERE clause to filter rows (e.g., "span.status_code = 'ERROR'").
            columns: Optional list of column names to include. If None, all columns are returned.
            stream_chunk_size: Optional chunk size for streaming large result sets.

        Raises:
            RuntimeError: If the Flight client request fails or returns no response.

        Notes:
            - Uses Apache Arrow Flight for efficient data transfer
            - Data is written directly to the specified path as a Parquet file
            - Large exports may benefit from specifying stream_chunk_size
        """
        with ArizeFlightClient(
            api_key=self._sdk_config.api_key,
            host=self._sdk_config.flight_host,
            port=self._sdk_config.flight_port,
            scheme=self._sdk_config.flight_scheme,
            request_verify=self._sdk_config.request_verify,
            max_chunksize=self._sdk_config.pyarrow_max_chunksize,
        ) as flight_client:
            exporter = ArizeExportClient(
                flight_client=flight_client,
            )
            exporter.export_to_parquet(
                path=path,
                space_id=space_id,
                model_id=project_name,
                environment=Environments.TRACING,
                start_time=start_time,
                end_time=end_time,
                where=where,
                columns=columns,
                stream_chunk_size=stream_chunk_size,
            )


def _build_patch_document(row: pd.Series) -> dict[str, object]:
    """Build a patch document from a pandas Series row by extracting metadata fields.

    Args:
        row: A pandas Series representing a row of data with potential metadata columns.

    Returns:
        dict[str, object]: A dictionary mapping metadata field names (without the
            'attributes.metadata.' prefix) to their values, preserving arrays and scalars.
    """
    # Extract and preserve metadata values with proper types
    patch = {}
    for key in row.index:
        if key.startswith("attributes.metadata."):
            field_name = key.replace("attributes.metadata.", "")
            # Check if the value is an array/list or other iterable (except strings)
            if isinstance(row[key], (list, np.ndarray)) or (
                hasattr(row[key], "__iter__") and not isinstance(row[key], str)
            ):
                # For arrays/iterables, just add the value (nulls will be handled later)
                patch[field_name] = row[key]
            else:
                # For scalar values, include even if it's None/null
                # This is important for explicitly setting fields to null
                patch[field_name] = row[key]
    return patch


def _process_patch_document(
    metadata_df: pd.DataFrame,
    patch_document_column_name: str,
    field_patches: pd.Series[Any],
    row_idx: int,
) -> dict[str, object]:
    """Process and merge patch documents from field patches and explicit patch column.

    Args:
        metadata_df: DataFrame containing the metadata with patch documents.
        patch_document_column_name: Name of the column containing explicit patch documents.
        field_patches: DataFrame containing patches derived from individual metadata fields.
        row_idx: The row index to process.

    Returns:
        dict[str, object]: Merged patch document where explicit patches take precedence over
            field patches. Returns empty dict if patch document is invalid or missing.
    """
    # Get the field patch for this row
    field_patch = field_patches.iloc[row_idx]

    # Get and process the explicit patch document
    patch_doc = metadata_df.loc[row_idx, patch_document_column_name]

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
    return {**field_patch, **explicit_patch}


def _ensure_dict_patch(
    metadata_df: pd.DataFrame,
    final_patch_column: str,
    row_idx: int,
) -> tuple[dict[str, object], list[ValidationError]]:
    """Ensure a patch value is a dictionary, converting from JSON string if needed.

    Args:
        metadata_df: DataFrame containing the patch data.
        final_patch_column: Name of the column containing the final patch document.
        row_idx: The row index to process.

    Returns:
        tuple[dict[str, object], list[ValidationError]]: A tuple containing:
            - The patch as a dictionary (empty dict if invalid or missing)
            - List of validation errors (empty if no errors)
    """
    patch = metadata_df.loc[row_idx, final_patch_column]
    validation_errors: list[ValidationError] = []

    # For None/null values, return an empty dict
    if patch is None:
        return {}, validation_errors

    # Handle NaN differently from None
    if isinstance(patch, float) and np.isnan(patch):
        return {}, validation_errors

    # If already a dict, return as is
    if isinstance(patch, dict):
        return patch, validation_errors

    # If string, try to parse as JSON
    if isinstance(patch, str):
        try:
            parsed = json.loads(patch)
            if isinstance(parsed, dict):
                return parsed, validation_errors
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in patch document: {e}"
            logger.warning(f"Row {row_idx}: {error_msg}")
            validation_errors.append(
                InvalidPatchDocumentFormat(row_idx, error_msg)
            )
            return {}, validation_errors  # if not validate else None
        else:
            error_msg = f"JSON must be an object/dictionary, got {type(parsed).__name__}"
            logger.warning(f"Row {row_idx}: {error_msg}")
            validation_errors.append(
                InvalidPatchDocumentFormat(row_idx, error_msg)
            )
            return {}, validation_errors  # if not validate else None

    # For other types, log warning
    error_msg = f"Unsupported patch type: {type(patch).__name__}"
    logger.warning(f"Row {row_idx}: {error_msg}")
    validation_errors.append(InvalidPatchDocumentFormat(row_idx, error_msg))
    return {}, validation_errors  # if not validate else None


def _format_note_for_storage(
    note_text: str,
    current_time_ms: int,
) -> list[str] | None:
    """Format a note text into a JSON-serialized list for storage.

    Args:
        note_text: The note text content to format.
        current_time_ms: The current timestamp in milliseconds.

    Returns:
        list[str] | None: A list containing a single JSON string with note metadata
            (text, updated_by, updated_at), or None if note_text is NaN/missing.
    """
    if pd.isna(note_text):
        return None
    note_obj = {
        "text": str(note_text),
        "updated_by": "SDK",
        "updated_at": current_time_ms,
    }
    return [json.dumps(note_obj)]


def _log_flight_update_summary(
    project_name: str,
    total_spans: int,
    request_type: FlightRequestType,
    response: FlightPostArrowFileResponse,
) -> None:
    """Log a structured summary of Flight update results including metrics and errors.

    Args:
        project_name: Name of the project being updated.
        total_spans: Total number of spans in the update request.
        request_type: The type of Flight request being performed.
        response: The Flight response object containing update results and errors.

    Notes:
        Logs one summary line with aggregated metrics, plus individual error lines
        for any failed span updates. Metrics include success rate, spans processed,
        and failure counts.
    """
    spans_updated = getattr(response, "spans_updated", None)
    if spans_updated is None:
        # Fallback for older response types
        spans_updated = getattr(response, "records_updated", None)
    spans_processed = getattr(response, "spans_processed", None)
    raw_errors = getattr(response, "errors", None)
    errors = (
        [
            {"span_id": e.span_id, "error_message": e.error_message}
            for e in raw_errors
        ]
        if raw_errors
        else []
    )

    # Normalize request_type to a readable string
    req_type_str = getattr(request_type, "name", None) or str(request_type)

    # Compute metrics safely
    success_rate = None
    spans_failed = None
    if isinstance(spans_processed, (int, float)) and spans_processed:
        su = int(spans_updated or 0)
        sp = int(spans_processed)
        success_rate = round(100.0 * su / sp, 2)
        spans_failed = max(sp - su, 0)

    metrics = {
        "project": project_name,
        "request_type": req_type_str,
        "total_spans": int(total_spans),
        "spans_processed": spans_processed,
        "spans_updated": spans_updated,
        "spans_failed": spans_failed,
        "success_rate": success_rate,
        "error_count": len(errors),
    }

    # One summary log line (great for JSON pipelines, readable in pretty mode)
    if spans_processed is None or spans_updated is None:
        logger.warning("Flight update response missing counts", extra=metrics)
    else:
        all_processed = int(spans_processed) == int(total_spans)
        msg = "All spans processed" if all_processed else "Partial processing"
        logger.info(msg, extra=metrics)

    # Emit individual error lines (structured per-error, easy to aggregate)
    for err in errors:
        logger.error(
            "Span update error",
            extra={
                "project": project_name,
                "request_type": req_type_str,
                **err,
            },
        )


def _message_to_dict(
    msg: message.Message,
    preserve_names: bool = True,
    use_int_enums: bool = False,
) -> dict[str, object]:
    """Convert a protobuf Message to a dictionary representation.

    Args:
        msg: The protobuf Message to convert.
        preserve_names: If True, preserve original proto field names. If False, use
            lowerCamelCase names. Defaults to True.
        use_int_enums: If True, represent enum values as integers. If False, use
            enum string names. Defaults to False.

    Returns:
        dict[str, object]: Dictionary representation of the protobuf message.
    """
    return json_format.MessageToDict(
        msg,
        preserving_proto_field_name=preserve_names,
        use_integers_for_enums=use_int_enums,
    )
