from datetime import datetime, timedelta
from itertools import chain
from typing import List, Optional, Union

import arize.pandas.tracing.columns as tracing_cols
import arize.pandas.tracing.constants as tracing_constants
import pandas as pd
import pyarrow as pa
from arize.pandas.tracing.types import StatusCodes
from arize.pandas.tracing.validation import errors as tracing_err
from arize.pandas.validation import errors as err
from arize.utils.constants import (
    MAX_EMBEDDING_DIMENSIONALITY,
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
)
from arize.utils.logging import logger
from arize.utils.types import is_dict_of, is_json_str


def validate_values(
    dataframe: pd.DataFrame,
    model_id: str,
    model_version: Optional[str] = None,
) -> List[err.ValidationError]:
    return list(
        chain(
            _check_invalid_model_id(model_id),
            _check_invalid_model_version(model_version),
            _check_span_root_field_values(dataframe),
            _check_span_attributes_values(dataframe),
        )
    )


def _check_invalid_model_id(model_id: Optional[str]) -> List[err.InvalidModelId]:
    # assume it's been coerced to string beforehand
    if (not isinstance(model_id, str)) or len(model_id.strip()) == 0:
        return [err.InvalidModelId()]
    return []


def _check_invalid_model_version(
    model_version: Optional[str] = None,
) -> List[err.InvalidModelVersion]:
    if model_version is None:
        return []
    if not isinstance(model_version, str) or len(model_version.strip()) == 0:
        return [err.InvalidModelVersion()]

    return []


def _check_span_root_field_values(
    dataframe: pd.DataFrame,
) -> List[err.ValidationError]:
    return list(
        chain(
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_SPAN_ID_COL.name,
                min_len=tracing_constants.SPAN_ID_MIN_STR_LENGTH,
                max_len=tracing_constants.SPAN_ID_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_SPAN_ID_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_TRACE_ID_COL.name,
                min_len=tracing_constants.SPAN_ID_MIN_STR_LENGTH,
                max_len=tracing_constants.SPAN_ID_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_TRACE_ID_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_PARENT_SPAN_ID_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_ID_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_PARENT_SPAN_ID_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_NAME_COL.name,
                min_len=tracing_constants.SPAN_NAME_MIN_STR_LENGTH,
                max_len=tracing_constants.SPAN_NAME_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_NAME_COL.required,
            ),
            _check_string_column_allowed_values(
                df=dataframe,
                col_name=tracing_cols.SPAN_STATUS_CODE_COL.name,
                allowed_values=StatusCodes.list_codes(),
                is_required=tracing_cols.SPAN_STATUS_CODE_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_STATUS_MESSAGE_COL.name,
                min_len=tracing_constants.SPAN_STATUS_MSG_MIN_STR_LENGTH,
                max_len=tracing_constants.SPAN_STATUS_MSG_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_STATUS_MESSAGE_COL.required,
            ),
            _check_value_columns_start_end_time(
                df=dataframe,
            ),
            _check_event_column_value(df=dataframe),
        )
    )


def _check_span_attributes_values(
    dataframe: pd.DataFrame,
) -> List[err.ValidationError]:
    return list(
        chain(
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_KIND_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_ID_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_KIND_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_EXCEPTION_MESSAGE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_EXCEPTION_MESSAGE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_EXCEPTION_MESSAGE_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_EXCEPTION_STACKTRACE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_EXCEPTION_STACK_TRACE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_EXCEPTION_STACKTRACE_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_INPUT_VALUE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_IO_VALUE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_INPUT_VALUE_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_OUTPUT_VALUE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_IO_VALUE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_OUTPUT_VALUE_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_INPUT_MIME_TYPE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_IO_MIME_TYPE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_INPUT_MIME_TYPE_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_OUTPUT_MIME_TYPE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_IO_MIME_TYPE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_OUTPUT_MIME_TYPE_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_EMBEDDING_MODEL_NAME_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_EMBEDDING_NAME_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_EMBEDDING_MODEL_NAME_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_LLM_MODEL_NAME_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_LLM_MODEL_NAME_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_LLM_MODEL_NAME_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_TEMPLATE_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_LLM_PROMPT_TEMPLATE_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_TEMPLATE_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_VERSION_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_LLM_PROMPT_TEMPLATE_VERSION_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_VERSION_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_TOOL_NAME_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_TOOL_NAME_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_TOOL_NAME_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_TOOL_DESCRIPTION_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_TOOL_DESCRIPTION_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_TOOL_DESCRIPTION_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_RERANKER_QUERY_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_RERANKER_QUERY_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_RERANKER_QUERY_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_RERANKER_MODEL_NAME_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_RERANKER_MODEL_NAME_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_RERANKER_MODEL_NAME_COL.required,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL.name,
                min_len=0,
                max_len=tracing_constants.JSON_STRING_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL.required,
                must_be_json=True,
            ),
            _check_string_column_value_length(
                df=dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_TOOL_PARAMETERS_COL.name,
                min_len=0,
                max_len=tracing_constants.SPAN_TOOL_PARAMETERS_MAX_STR_LENGTH,
                is_required=tracing_cols.SPAN_ATTRIBUTES_TOOL_PARAMETERS_COL.required,
                must_be_json=True,
            ),
            _check_embeddings_column_value(dataframe),
            _check_LLM_IO_messages_column_value(
                dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL.name,
            ),
            _check_LLM_IO_messages_column_value(
                dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_LLM_OUTPUT_MESSAGES_COL.name,
            ),
            _check_documents_column_value(
                dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_RETRIEVAL_DOCUMENTS_COL.name,
            ),
            _check_documents_column_value(
                dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_RERANKER_INPUT_DOCUMENTS_COL.name,
            ),
            _check_documents_column_value(
                dataframe,
                col_name=tracing_cols.SPAN_ATTRIBUTES_RERANKER_OUTPUT_DOCUMENTS_COL.name,
            ),
        )
    )


def _check_string_column_value_length(
    df: pd.DataFrame,
    col_name: str,
    min_len: int,
    max_len: int,
    is_required: bool,
    must_be_json: bool = False,
) -> List[Union[tracing_err.InvalidMissingValueInColumn, tracing_err.InvalidStringLengthInColumn]]:
    if col_name not in df.columns:
        return []

    errors = []
    if is_required and df[col_name].isnull().any():
        errors.append(
            tracing_err.InvalidMissingValueInColumn(
                col_name=col_name,
            )
        )

    if not (
        # Check that the non-None values of the desired colum have a
        # string length between min_len and max_len
        # Does not check the None values
        df[~df[col_name].isnull()][col_name]
        .astype(str)
        .str.len()
        .between(min_len, max_len)
        .all()
    ):
        errors.append(
            tracing_err.InvalidStringLengthInColumn(
                col_name=col_name,
                min_length=min_len,
                max_length=max_len,
            )
        )
    if must_be_json and not df[~df[col_name].isnull()][col_name].apply(is_json_str).all():
        errors.append(tracing_err.InvalidJsonStringInColumn(col_name=col_name))

    return errors


def _check_string_column_allowed_values(
    df: pd.DataFrame,
    col_name: str,
    allowed_values: List[str],
    is_required: bool,
) -> List[
    Union[tracing_err.InvalidMissingValueInColumn, tracing_err.InvalidStringValueNotAllowedInColumn]
]:
    if col_name not in df.columns:
        return []

    errors = []
    if is_required and df[col_name].isnull().any():
        errors.append(
            tracing_err.InvalidMissingValueInColumn(
                col_name=col_name,
            )
        )

    # We compare in lowercase
    allowed_values_lowercase = [v.lower() for v in allowed_values]
    if not (
        # Check that the non-None values of the desired colum have a
        # string values amongst the ones allowed
        # Does not check the None values
        df[~df[col_name].isnull()][col_name]
        .astype(str)
        .str.lower()
        .isin(allowed_values_lowercase)
        .all()
    ):
        errors.append(
            tracing_err.InvalidStringValueNotAllowedInColumn(
                col_name=col_name,
                allowed_values=allowed_values,
            )
        )
    return errors


def _check_value_columns_start_end_time(
    df: pd.DataFrame,
) -> List[
    Union[
        tracing_err.InvalidMissingValueInColumn,
        tracing_err.InvalidTimestampValueInColumn,
        tracing_err.InvalidStartAndEndTimeValuesInColumn,
    ]
]:
    errors = []
    errors += _check_value_timestamp(
        df=df,
        col_name=tracing_cols.SPAN_START_TIME_COL.name,
        is_required=tracing_cols.SPAN_START_TIME_COL.required,
    )
    errors += _check_value_timestamp(
        df=df,
        col_name=tracing_cols.SPAN_END_TIME_COL.name,
        is_required=tracing_cols.SPAN_END_TIME_COL.required,
    )
    if (
        tracing_cols.SPAN_START_TIME_COL.name in df.columns
        and tracing_cols.SPAN_END_TIME_COL.name in df.columns
        and (
            df[tracing_cols.SPAN_START_TIME_COL.name] > df[tracing_cols.SPAN_END_TIME_COL.name]
        ).any()
    ):
        errors.append(
            tracing_err.InvalidStartAndEndTimeValuesInColumn(
                greater_col_name=tracing_cols.SPAN_END_TIME_COL.name,
                less_col_name=tracing_cols.SPAN_START_TIME_COL.name,
            )
        )
    return errors


def _check_value_timestamp(
    df: pd.DataFrame,
    col_name: str,
    is_required: bool,
) -> List[
    Union[tracing_err.InvalidMissingValueInColumn, tracing_err.InvalidTimestampValueInColumn]
]:
    # This check expects that timestamps have previously been converted to nanoseconds
    if col_name not in df.columns:
        return []

    errors = []
    if is_required and df[col_name].isnull().any():
        errors.append(
            tracing_err.InvalidMissingValueInColumn(
                col_name=col_name,
            )
        )

    now_t = datetime.now()
    lbound, ubound = (
        (now_t - timedelta(days=MAX_PAST_YEARS_FROM_CURRENT_TIME * 365)).timestamp() * 1e9,
        (now_t + timedelta(days=MAX_FUTURE_YEARS_FROM_CURRENT_TIME * 365)).timestamp() * 1e9,
    )

    # faster than pyarrow compute
    stats = df[col_name].agg(["min", "max"])

    ta = pa.Table.from_pandas(stats.to_frame())
    min_, max_ = ta.column(0)
    if max_.as_py() > now_t.timestamp() * 1e9:
        logger.warning(
            f"Detected future timestamp in column '{col_name}'. "
            "Caution when sending spans with future timestamps. "
            "Arize only stores 2 years worth of data. For example, if you sent spans "
            "to Arize from 1.5 years ago, and now send spans with timestamps of a year in "
            "the future, the oldest 0.5 years will be dropped to maintain the 2 years worth of data "
            "requirement."
        )

    if min_.as_py() < lbound or max_.as_py() > ubound:
        return [tracing_err.InvalidTimestampValueInColumn(timestamp_col_name=col_name)]

    return []


def _check_event_column_value(
    df: pd.DataFrame,
) -> List[tracing_err.InvalidEventValueInColumn]:
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
            if attrs and not wrong_attrs_found and not is_dict_of(attrs, key_allowed_types=str):
                wrong_attrs_found = True
        if wrong_name_found and wrong_attrs_found:
            break

    if wrong_name_found or wrong_attrs_found:
        return [
            tracing_err.InvalidEventValueInColumn(
                col_name=col_name,
                wrong_name=wrong_name_found,
                wrong_time=False,
                wrong_attrs=wrong_attrs_found,
            )
        ]
    return []


def _check_embeddings_column_value(
    df: pd.DataFrame,
) -> List[tracing_err.InvalidEmbeddingValueInColumn]:
    col_name = tracing_cols.SPAN_ATTRIBUTES_EMBEDDING_EMBEDDINGS_COL.name
    if col_name not in df.columns:
        return []

    wrong_vector_found, wrong_text_found = False, False
    for row in df[~df[col_name].isnull()][col_name]:
        for emb_object in row:
            # validate vector
            vector = emb_object.get(tracing_cols.SPAN_ATTRIBUTES_EMBEDDING_VECTOR_KEY)
            if (
                vector is not None
                and not wrong_vector_found
                and (len(vector) > MAX_EMBEDDING_DIMENSIONALITY or len(vector) == 1)
            ):
                wrong_vector_found = True
            # validate text
            text = emb_object.get(tracing_cols.SPAN_ATTRIBUTES_EMBEDDING_TEXT_KEY)
            if text and len(text) > tracing_constants.SPAN_EMBEDDING_TEXT_MAX_STR_LENGTH:
                wrong_text_found = True
        if wrong_vector_found and wrong_text_found:
            break

    if wrong_vector_found or wrong_text_found:
        return [
            tracing_err.InvalidEmbeddingValueInColumn(
                col_name=col_name,
                wrong_vector=wrong_vector_found,
                wrong_text=wrong_text_found,
            )
        ]
    return []


def _check_LLM_IO_messages_column_value(
    df: pd.DataFrame,
    col_name: str,
) -> List[tracing_err.InvalidLLMMessageValueInColumn]:
    if col_name not in df.columns:
        return []

    wrong_role_found, wrong_content_found, wrong_tool_calls_found = False, False, False
    for row in df[~df[col_name].isnull()][col_name]:
        for message in row:
            # validate role
            role = message.get(tracing_cols.SPAN_ATTRIBUTES_MESSAGE_ROLE_KEY)
            if (
                role
                and not wrong_role_found
                and len(role) > tracing_constants.SPAN_LLM_MESSAGE_ROLE_MAX_STR_LENGTH
            ):
                wrong_role_found = True
            # validate content
            content = message.get(tracing_cols.SPAN_ATTRIBUTES_MESSAGE_CONTENT_KEY)
            if (
                content
                and not wrong_content_found
                and len(content) > tracing_constants.SPAN_LLM_MESSAGE_CONTENT_MAX_STR_LENGTH
            ):
                wrong_content_found = True
            # validate tool calls
            tool_calls = message.get(tracing_cols.SPAN_ATTRIBUTES_MESSAGE_TOOL_CALLS_KEY)
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
                        len(function_args) > tracing_constants.JSON_STRING_MAX_STR_LENGTH
                        or not is_json_str(function_args)
                    ):
                        wrong_tool_calls_found = True
                        break
        if wrong_role_found and wrong_content_found and wrong_tool_calls_found:
            break

    if wrong_role_found or wrong_content_found or wrong_tool_calls_found:
        return [
            tracing_err.InvalidLLMMessageValueInColumn(
                col_name=col_name,
                wrong_role=wrong_role_found,
                wrong_content=wrong_content_found,
                wrong_tool_calls=wrong_tool_calls_found,
            )
        ]
    return []


def _check_documents_column_value(
    df: pd.DataFrame,
    col_name: str,
) -> List[tracing_err.InvalidDocumentValueInColumn]:
    if col_name not in df.columns:
        return []

    wrong_id_found, wrong_content_found, wrong_metadata_found = False, False, False
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
                and len(content) > tracing_constants.SPAN_DOCUMENT_CONTENT_MAX_STR_LENGTH
            ):
                wrong_content_found = True
            # validate metadata
            metadata = doc.get(tracing_cols.SPAN_ATTRIBUTES_DOCUMENT_METADATA_KEY)
            if (
                metadata
                and not wrong_metadata_found
                and (
                    len(metadata) > tracing_constants.JSON_STRING_MAX_STR_LENGTH
                    or not is_json_str(metadata)
                )
            ):
                wrong_metadata_found = True
        if wrong_id_found and wrong_content_found and wrong_metadata_found:
            break

    if wrong_id_found or wrong_content_found or wrong_metadata_found:
        return [
            tracing_err.InvalidDocumentValueInColumn(
                col_name=col_name,
                wrong_id=wrong_id_found,
                wrong_content=wrong_content_found,
                wrong_metadata=wrong_metadata_found,
            )
        ]
    return []
