"""Value validation logic for span data."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

from arize.constants import spans as tracing_constants
from arize.constants.ml import MAX_EMBEDDING_DIMENSIONALITY
from arize.ml.types import StatusCodes
from arize.spans import columns as tracing_cols
from arize.spans.validation.common import value_validation
from arize.spans.validation.common.errors import (
    InvalidDocumentValueInColumn,
    InvalidEmbeddingValueInColumn,
    InvalidEventValueInColumn,
    InvalidLLMMessageValueInColumn,
)
from arize.utils.types import is_dict_of, is_json_str

if TYPE_CHECKING:
    import pandas as pd

    from arize.exceptions.base import ValidationError


def check_span_root_field_values(
    dataframe: pd.DataFrame,
) -> list[ValidationError]:
    """Validates root-level span field values for proper format and constraints.

    Checks span ID, trace ID, parent span ID, name, status code, status message,
    timestamps, and events for conformance to specification limits.

    Args:
        dataframe: The DataFrame containing span data.

    Returns:
        List of validation errors found in root span fields.
    """
    return list(
        chain(
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_SPAN_ID_COL.name,
                min_len=tracing_constants.SPAN_ID_MIN_STR_LENGTH,
                max_len=tracing_constants.SPAN_ID_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_SPAN_ID_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_TRACE_ID_COL.name,
                min_len=tracing_constants.SPAN_ID_MIN_STR_LENGTH,
                max_len=tracing_constants.SPAN_ID_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_TRACE_ID_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_PARENT_SPAN_ID_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_ID_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_PARENT_SPAN_ID_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_NAME_COL.name,
                min_len=tracing_constants.SPAN_NAME_MIN_STR_LENGTH,
                max_len=tracing_constants.SPAN_NAME_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_NAME_COL.required,
            ),
            value_validation.check_string_column_allowed_values(
                df=dataframe,
                col_name=tracing_cols.SPAN_STATUS_CODE_COL.name,
                allowed_values=StatusCodes.list_codes(),
                is_required=tracing_cols.SPAN_STATUS_CODE_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_STATUS_MESSAGE_COL.name,
                min_len=tracing_constants.SPAN_STATUS_MSG_MIN_STR_LENGTH,
                max_len=tracing_constants.SPAN_STATUS_MSG_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_STATUS_MESSAGE_COL.required,
            ),
            value_validation.check_value_columns_start_end_time(
                df=dataframe,
            ),
            check_event_column_value(df=dataframe),
        )
    )


def check_span_attributes_values(
    dataframe: pd.DataFrame,
) -> list[ValidationError]:
    """Validates span attribute values for proper format and constraints.

    Checks all span attributes including LLM parameters, embeddings, documents,
    tools, and other metadata fields for conformance to specification limits.

    Args:
        dataframe: The DataFrame containing span data.

    Returns:
        List of validation errors found in span attributes.
    """
    return list(
        chain(
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_KIND_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_ID_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_KIND_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_EXCEPTION_MESSAGE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_EXCEPTION_MESSAGE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_EXCEPTION_MESSAGE_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_EXCEPTION_STACKTRACE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_EXCEPTION_STACK_TRACE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_EXCEPTION_STACKTRACE_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_INPUT_VALUE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_IO_VALUE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_INPUT_VALUE_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_OUTPUT_VALUE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_IO_VALUE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_OUTPUT_VALUE_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_INPUT_MIME_TYPE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_IO_MIME_TYPE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_INPUT_MIME_TYPE_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_OUTPUT_MIME_TYPE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_IO_MIME_TYPE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_OUTPUT_MIME_TYPE_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_EMBEDDING_MODEL_NAME_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_EMBEDDING_NAME_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_EMBEDDING_MODEL_NAME_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_LLM_MODEL_NAME_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_LLM_MODEL_NAME_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_LLM_MODEL_NAME_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_TEMPLATE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_LLM_PROMPT_TEMPLATE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_TEMPLATE_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_VERSION_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_LLM_PROMPT_TEMPLATE_VERSION_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_VERSION_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_TOOL_NAME_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_TOOL_NAME_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_TOOL_NAME_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_TOOL_DESCRIPTION_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_TOOL_DESCRIPTION_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_TOOL_DESCRIPTION_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_RERANKER_QUERY_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_RERANKER_QUERY_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_RERANKER_QUERY_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_RERANKER_MODEL_NAME_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_RERANKER_MODEL_NAME_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_RERANKER_MODEL_NAME_COL.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL.name,
                min_len=0,
                max_len=tracing_constants.JSON_STRING_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL.required,
                must_be_json=True,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_TOOL_PARAMETERS_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_TOOL_PARAMETERS_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_TOOL_PARAMETERS_COL.required,
                must_be_json=True,
            ),
            check_embeddings_column_value(dataframe),
            check_LLM_IO_messages_column_value(
                dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL.name,
            ),
            check_LLM_IO_messages_column_value(
                dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_LLM_OUTPUT_MESSAGES_COL.name,
            ),
            check_documents_column_value(
                dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_RETRIEVAL_DOCUMENTS_COL.name,
            ),
            check_documents_column_value(
                dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_RERANKER_INPUT_DOCUMENTS_COL.name,
            ),
            check_documents_column_value(
                dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_RERANKER_OUTPUT_DOCUMENTS_COL.name,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_SESSION_ID.name,
                min_len=0,
                max_len=tracing_constants.SESSION_ID_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_SESSION_ID.required,
            ),
            value_validation.check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_USER_ID.name,
                min_len=0,
                max_len=tracing_constants.USER_ID_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_USER_ID.required,
            ),
        )
    )


def check_event_column_value(
    df: pd.DataFrame,
) -> list[InvalidEventValueInColumn]:
    """Validates span event column values for proper format and length constraints.

    Checks event names for length limits and attributes for proper dictionary structure.

    Args:
        df: The DataFrame containing span events.

    Returns:
        List of validation errors found in event column values.
    """
    col_name = tracing_cols.SPAN_EVENTS_COL.name
    if col_name not in df.columns:
        return []

    # NOTE: We are leaving out timestamp validation until we learn more
    wrong_name_found, wrong_attrs_found = False, False
    for row in df[col_name]:
        for event in row:
            # validate name
            name = event.get(tracing_cols.SPAN_EVENT_NAME_KEY)
            if (
                name
                and not wrong_name_found
                and len(name) > tracing_constants.SPAN_EVENT_NAME_MAX_STR_LENGTH
            ):
                wrong_name_found = True
            # validate attributes
            attrs = event.get(tracing_cols.SPAN_EVENT_ATTRIBUTES_KEY)
            if (
                attrs
                and not wrong_attrs_found
                and not is_dict_of(attrs, key_allowed_types=str)
            ):
                wrong_attrs_found = True
        if wrong_name_found and wrong_attrs_found:
            break

    if wrong_name_found or wrong_attrs_found:
        return [
            InvalidEventValueInColumn(
                col_name=col_name,
                wrong_name=wrong_name_found,
                wrong_time=False,
                wrong_attrs=wrong_attrs_found,
            )
        ]
    return []


def check_embeddings_column_value(
    df: pd.DataFrame,
) -> list[InvalidEmbeddingValueInColumn]:
    """Validates embedding column values for proper vector dimensions and text length.

    Checks that embedding vectors are within dimensionality limits and text
    values don't exceed maximum length.

    Args:
        df: The DataFrame containing embedding data.

    Returns:
        List of validation errors found in embedding column values.
    """
    col_name = tracing_cols.SPAN_ATTRIBUTES_EMBEDDING_EMBEDDINGS_COL.name
    if col_name not in df.columns:
        return []

    wrong_vector_found, wrong_text_found = False, False
    for row in df[~df[col_name].isnull()][col_name]:
        for emb_object in row:
            # validate vector
            vector = emb_object.get(
                tracing_cols.SPAN_ATTRIBUTES_EMBEDDING_VECTOR_KEY
            )
            if (
                vector is not None
                and not wrong_vector_found
                and (
                    len(vector) > MAX_EMBEDDING_DIMENSIONALITY
                    or len(vector) == 1
                )
            ):
                wrong_vector_found = True
            # validate text
            text = emb_object.get(
                tracing_cols.SPAN_ATTRIBUTES_EMBEDDING_TEXT_KEY
            )
            if (
                text
                and len(text)
                > tracing_constants.SPAN_EMBEDDING_TEXT_MAX_STR_LENGTH
            ):
                wrong_text_found = True
        if wrong_vector_found and wrong_text_found:
            break

    if wrong_vector_found or wrong_text_found:
        return [
            InvalidEmbeddingValueInColumn(
                col_name=col_name,
                wrong_vector=wrong_vector_found,
                wrong_text=wrong_text_found,
            )
        ]
    return []


def check_LLM_IO_messages_column_value(
    df: pd.DataFrame,
    col_name: str,
) -> list[InvalidLLMMessageValueInColumn]:
    """Validates LLM input/output message column values for proper format and length.

    Checks message role, content, and tool calls for conformance to length limits
    and proper JSON formatting.

    Args:
        df: The DataFrame containing LLM messages.
        col_name: Name of the message column to validate.

    Returns:
        List of validation errors found in message column values.
    """
    if col_name not in df.columns:
        return []

    wrong_role_found, wrong_content_found, wrong_tool_calls_found = (
        False,
        False,
        False,
    )
    for row in df[~df[col_name].isnull()][col_name]:
        for message in row:
            # validate role
            role = message.get(tracing_cols.SPAN_ATTRIBUTES_MESSAGE_ROLE_KEY)
            if (
                role
                and not wrong_role_found
                and len(role)
                > tracing_constants.SPAN_LLM_MESSAGE_ROLE_MAX_STR_LENGTH
            ):
                wrong_role_found = True
            # validate content
            content = message.get(
                tracing_cols.SPAN_ATTRIBUTES_MESSAGE_CONTENT_KEY
            )
            if (
                content
                and not wrong_content_found
                and len(content)
                > tracing_constants.SPAN_LLM_MESSAGE_CONTENT_MAX_STR_LENGTH
            ):
                wrong_content_found = True
            # validate tool calls
            tool_calls = message.get(
                tracing_cols.SPAN_ATTRIBUTES_MESSAGE_TOOL_CALLS_KEY
            )
            if tool_calls and not wrong_tool_calls_found:
                for tc in tool_calls:
                    function_name = tc.get(
                        tracing_cols.SPAN_ATTRIBUTES_MESSAGE_TOOL_CALLS_FUNCTION_NAME_KEY
                    )
                    if (
                        function_name
                        and len(function_name)
                        > tracing_constants.SPAN_LLM_TOOL_CALL_FUNCTION_NAME_MAX_STR_LENGTH
                    ):
                        wrong_tool_calls_found = True
                        break
                    function_args = tc.get(
                        tracing_cols.SPAN_ATTRIBUTES_MESSAGE_TOOL_CALLS_FUNCTION_ARGUMENTS_KEY
                    )
                    if function_args and (
                        len(function_args)
                        > tracing_constants.JSON_STRING_MAX_STR_LENGTH
                        or not is_json_str(function_args)
                    ):
                        wrong_tool_calls_found = True
                        break
        if wrong_role_found and wrong_content_found and wrong_tool_calls_found:
            break

    if wrong_role_found or wrong_content_found or wrong_tool_calls_found:
        return [
            InvalidLLMMessageValueInColumn(
                col_name=col_name,
                wrong_role=wrong_role_found,
                wrong_content=wrong_content_found,
                wrong_tool_calls=wrong_tool_calls_found,
            )
        ]
    return []


def check_documents_column_value(
    df: pd.DataFrame,
    col_name: str,
) -> list[InvalidDocumentValueInColumn]:
    """Validates document column values for proper format and length constraints.

    Checks document ID, content, and metadata for conformance to length limits
    and proper data type requirements.

    Args:
        df: The DataFrame containing documents.
        col_name: Name of the document column to validate.

    Returns:
        List of validation errors found in document column values.
    """
    if col_name not in df.columns:
        return []

    wrong_id_found, wrong_content_found, wrong_metadata_found = (
        False,
        False,
        False,
    )
    for row in df[~df[col_name].isnull()][col_name]:
        for doc in row:
            # validate id
            id = doc.get(tracing_cols.SPAN_ATTRIBUTES_DOCUMENT_ID_KEY)
            if (
                id
                and not wrong_id_found
                and len(id) > tracing_constants.SPAN_DOCUMENT_ID_MAX_STR_LENGTH
            ):
                wrong_id_found = True
            # validate content
            content = doc.get(tracing_cols.SPAN_ATTRIBUTES_DOCUMENT_CONTENT_KEY)
            if (
                content
                and not wrong_content_found
                and len(content)
                > tracing_constants.SPAN_DOCUMENT_CONTENT_MAX_STR_LENGTH
            ):
                wrong_content_found = True
            # validate metadata
            metadata = doc.get(
                tracing_cols.SPAN_ATTRIBUTES_DOCUMENT_METADATA_KEY
            )
            if (
                metadata
                and not wrong_metadata_found
                and not is_dict_of(
                    d=metadata,
                    key_allowed_types=str,
                    value_allowed_types=(bool, str, int, float),
                    value_list_allowed_types=(bool, str, int, float),
                )
            ):
                wrong_metadata_found = True
        if wrong_id_found and wrong_content_found and wrong_metadata_found:
            break

    if wrong_id_found or wrong_content_found or wrong_metadata_found:
        return [
            InvalidDocumentValueInColumn(
                col_name=col_name,
                wrong_id=wrong_id_found,
                wrong_content=wrong_content_found,
                wrong_metadata=wrong_metadata_found,
            )
        ]
    return []
