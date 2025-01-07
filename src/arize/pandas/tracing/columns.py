from enum import Enum

import openinference.semconv.trace as oinf
import opentelemetry.semconv.trace as otel


class SpanColumnDataType(Enum):
    BOOL = 1
    NUMERIC = 2
    STRING = 3
    JSON = 4
    TIMESTAMP = 5
    DICT = 6
    LIST_DICT = 7


class SpanColumn:
    def __init__(
        self,
        name: str,
        data_type: SpanColumnDataType,
        required: bool = False,
    ) -> None:
        self.name = name
        self.required = required
        self.data_type = data_type


# Root level columns
SPAN_TRACE_ID_COL = SpanColumn(
    name="context.trace_id",
    data_type=SpanColumnDataType.STRING,
    required=True,
)
SPAN_SPAN_ID_COL = SpanColumn(
    name="context.span_id",
    data_type=SpanColumnDataType.STRING,
    required=True,
)
SPAN_PARENT_SPAN_ID_COL = SpanColumn(
    name="parent_id",
    data_type=SpanColumnDataType.STRING,
)
SPAN_NAME_COL = SpanColumn(
    name="name",
    data_type=SpanColumnDataType.STRING,
    required=True,
)
SPAN_START_TIME_COL = SpanColumn(
    name="start_time",
    data_type=SpanColumnDataType.TIMESTAMP,
)
SPAN_END_TIME_COL = SpanColumn(
    name="end_time",
    data_type=SpanColumnDataType.TIMESTAMP,
)
# Status columns
SPAN_STATUS_CODE_COL = SpanColumn(
    name="status_code",
    data_type=SpanColumnDataType.STRING,
)
SPAN_STATUS_MESSAGE_COL = SpanColumn(
    name="status_message",
    data_type=SpanColumnDataType.STRING,
)
# Events columns
SPAN_EVENTS_COL = SpanColumn(
    name="events",
    data_type=SpanColumnDataType.LIST_DICT,
)
SPAN_EVENT_NAME_KEY = "name"
SPAN_EVENT_TIME_KEY = "timestamp"
SPAN_EVENT_ATTRIBUTES_KEY = "attributes"
# Attributes
ROOT_LEVEL_SPAN_KIND_COL = SpanColumn(
    name="span_kind",
    data_type=SpanColumnDataType.STRING,
)
SPAN_KIND_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.OPENINFERENCE_SPAN_KIND}",
    data_type=SpanColumnDataType.STRING,
)
# Attributes Exception columns
SPAN_ATTRIBUTES_EXCEPTION_TYPE_COL = SpanColumn(
    name=f"attributes.{otel.SpanAttributes.EXCEPTION_TYPE}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_EXCEPTION_MESSAGE_COL = SpanColumn(
    name=f"attributes.{otel.SpanAttributes.EXCEPTION_MESSAGE}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_EXCEPTION_ESCAPED_COL = SpanColumn(
    name=f"attributes.{otel.SpanAttributes.EXCEPTION_ESCAPED}",
    data_type=SpanColumnDataType.BOOL,
)
SPAN_ATTRIBUTES_EXCEPTION_STACKTRACE_COL = SpanColumn(
    name=f"attributes.{otel.SpanAttributes.EXCEPTION_STACKTRACE}",
    data_type=SpanColumnDataType.STRING,
)
# Attributes Input columns
SPAN_ATTRIBUTES_INPUT_VALUE_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.INPUT_VALUE}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_INPUT_MIME_TYPE_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.INPUT_MIME_TYPE}",
    data_type=SpanColumnDataType.STRING,
)
# Attributes Output columns
SPAN_ATTRIBUTES_OUTPUT_VALUE_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.OUTPUT_VALUE}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_OUTPUT_MIME_TYPE_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.OUTPUT_MIME_TYPE}",
    data_type=SpanColumnDataType.STRING,
)
# Attributes Embedding columns
SPAN_ATTRIBUTES_EMBEDDING_MODEL_NAME_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.EMBEDDING_MODEL_NAME}",
    data_type=SpanColumnDataType.STRING,
)

SPAN_ATTRIBUTES_EMBEDDING_EMBEDDINGS_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.EMBEDDING_EMBEDDINGS}",
    data_type=SpanColumnDataType.LIST_DICT,
)
# Embedddings Keys
SPAN_ATTRIBUTES_EMBEDDING_VECTOR_KEY = (
    f"{oinf.EmbeddingAttributes.EMBEDDING_VECTOR}"
)
SPAN_ATTRIBUTES_EMBEDDING_TEXT_KEY = (
    f"{oinf.EmbeddingAttributes.EMBEDDING_TEXT}"
)
# Attributes LLM columns
SPAN_ATTRIBUTES_LLM_MODEL_NAME_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.LLM_MODEL_NAME}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.LLM_INPUT_MESSAGES}",
    data_type=SpanColumnDataType.LIST_DICT,
)
SPAN_ATTRIBUTES_LLM_OUTPUT_MESSAGES_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.LLM_OUTPUT_MESSAGES}",
    data_type=SpanColumnDataType.LIST_DICT,
)
SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.LLM_INVOCATION_PARAMETERS}",
    data_type=SpanColumnDataType.JSON,
)
SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_TEMPLATE_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.LLM_PROMPT_TEMPLATE}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_VARIABLES_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES}",
    data_type=SpanColumnDataType.DICT,
)
SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_VERSION_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_LLM_PROMPT_TOKEN_COUNT_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.LLM_TOKEN_COUNT_PROMPT}",
    data_type=SpanColumnDataType.NUMERIC,
)
SPAN_ATTRIBUTES_LLM_COMPLETION_TOKEN_COUNT_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.LLM_TOKEN_COUNT_COMPLETION}",
    data_type=SpanColumnDataType.NUMERIC,
)
SPAN_ATTRIBUTES_LLM_TOTAL_TOKEN_COUNT_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.LLM_TOKEN_COUNT_TOTAL}",
    data_type=SpanColumnDataType.NUMERIC,
)
# Attributes Message Keys
SPAN_ATTRIBUTES_MESSAGE_NAME_KEY = f"{oinf.MessageAttributes.MESSAGE_NAME}"
SPAN_ATTRIBUTES_MESSAGE_ROLE_KEY = f"{oinf.MessageAttributes.MESSAGE_ROLE}"
SPAN_ATTRIBUTES_MESSAGE_CONTENT_KEY = (
    f"{oinf.MessageAttributes.MESSAGE_CONTENT}"
)
SPAN_ATTRIBUTES_MESSAGE_TOOL_CALLS_KEY = (
    f"{oinf.MessageAttributes.MESSAGE_TOOL_CALLS}"
)
SPAN_ATTRIBUTES_MESSAGE_TOOL_CALLS_FUNCTION_NAME_KEY = (
    f"{oinf.ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
)
SPAN_ATTRIBUTES_MESSAGE_TOOL_CALLS_FUNCTION_ARGUMENTS_KEY = (
    f"{oinf.ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
)
# Attributes Tool columns
SPAN_ATTRIBUTES_TOOL_NAME_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.TOOL_NAME}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_TOOL_DESCRIPTION_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.TOOL_DESCRIPTION}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_TOOL_PARAMETERS_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.TOOL_PARAMETERS}",
    data_type=SpanColumnDataType.JSON,
)
# Attributes Retrieval columns
SPAN_ATTRIBUTES_RETRIEVAL_DOCUMENTS_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.RETRIEVAL_DOCUMENTS}",
    data_type=SpanColumnDataType.LIST_DICT,
)
# Document Object Keys
SPAN_ATTRIBUTES_DOCUMENT_ID_KEY = f"{oinf.DocumentAttributes.DOCUMENT_ID}"
SPAN_ATTRIBUTES_DOCUMENT_SCORE_KEY = f"{oinf.DocumentAttributes.DOCUMENT_SCORE}"
SPAN_ATTRIBUTES_DOCUMENT_CONTENT_KEY = (
    f"{oinf.DocumentAttributes.DOCUMENT_CONTENT}"
)
SPAN_ATTRIBUTES_DOCUMENT_METADATA_KEY = (
    f"{oinf.DocumentAttributes.DOCUMENT_METADATA}"
)
# Attributes Reranker columns
SPAN_ATTRIBUTES_RERANKER_INPUT_DOCUMENTS_COL = SpanColumn(
    name=f"attributes.{oinf.RerankerAttributes.RERANKER_INPUT_DOCUMENTS}",
    data_type=SpanColumnDataType.LIST_DICT,
)
SPAN_ATTRIBUTES_RERANKER_OUTPUT_DOCUMENTS_COL = SpanColumn(
    name=f"attributes.{oinf.RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS}",
    data_type=SpanColumnDataType.LIST_DICT,
)
SPAN_ATTRIBUTES_RERANKER_QUERY_COL = SpanColumn(
    name=f"attributes.{oinf.RerankerAttributes.RERANKER_QUERY}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_RERANKER_MODEL_NAME_COL = SpanColumn(
    name=f"attributes.{oinf.RerankerAttributes.RERANKER_MODEL_NAME}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_RERANKER_TOP_K_COL = SpanColumn(
    name=f"attributes.{oinf.RerankerAttributes.RERANKER_TOP_K}",
    data_type=SpanColumnDataType.NUMERIC,
)
SPAN_ATTRIBUTES_SESSION_ID = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.SESSION_ID}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_USER_ID = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.USER_ID}",
    data_type=SpanColumnDataType.STRING,
)
SPAN_ATTRIBUTES_METADATA = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.METADATA}",
    data_type=SpanColumnDataType.DICT,
)
SPAN_ATTRIBUTES_LLM_TOOLS_COL = SpanColumn(
    name=f"attributes.{oinf.SpanAttributes.LLM_TOOLS}",
    data_type=SpanColumnDataType.LIST_DICT,
)

# TODO(Kiko): Add a test that checks no dupes in following lists
SPAN_OPENINFERENCE_COLUMNS = [
    SPAN_TRACE_ID_COL,
    SPAN_SPAN_ID_COL,
    SPAN_PARENT_SPAN_ID_COL,
    ROOT_LEVEL_SPAN_KIND_COL,
    SPAN_KIND_COL,
    SPAN_NAME_COL,
    SPAN_START_TIME_COL,
    SPAN_END_TIME_COL,
    SPAN_STATUS_CODE_COL,
    SPAN_STATUS_MESSAGE_COL,
    SPAN_EVENTS_COL,
    SPAN_ATTRIBUTES_EXCEPTION_TYPE_COL,
    SPAN_ATTRIBUTES_EXCEPTION_MESSAGE_COL,
    SPAN_ATTRIBUTES_EXCEPTION_ESCAPED_COL,
    SPAN_ATTRIBUTES_EXCEPTION_STACKTRACE_COL,
    SPAN_ATTRIBUTES_INPUT_VALUE_COL,
    SPAN_ATTRIBUTES_INPUT_MIME_TYPE_COL,
    SPAN_ATTRIBUTES_OUTPUT_VALUE_COL,
    SPAN_ATTRIBUTES_OUTPUT_MIME_TYPE_COL,
    SPAN_ATTRIBUTES_EMBEDDING_EMBEDDINGS_COL,
    SPAN_ATTRIBUTES_LLM_INVOCATION_PARAMETERS_COL,
    SPAN_ATTRIBUTES_LLM_INPUT_MESSAGES_COL,
    SPAN_ATTRIBUTES_LLM_OUTPUT_MESSAGES_COL,
    SPAN_ATTRIBUTES_LLM_MODEL_NAME_COL,
    SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_TEMPLATE_COL,
    SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_VARIABLES_COL,
    SPAN_ATTRIBUTES_LLM_PROMPT_TEMPLATE_VERSION_COL,
    SPAN_ATTRIBUTES_LLM_PROMPT_TOKEN_COUNT_COL,
    SPAN_ATTRIBUTES_LLM_COMPLETION_TOKEN_COUNT_COL,
    SPAN_ATTRIBUTES_LLM_TOTAL_TOKEN_COUNT_COL,
    SPAN_ATTRIBUTES_TOOL_NAME_COL,
    SPAN_ATTRIBUTES_TOOL_DESCRIPTION_COL,
    SPAN_ATTRIBUTES_TOOL_PARAMETERS_COL,
    SPAN_ATTRIBUTES_RETRIEVAL_DOCUMENTS_COL,
    SPAN_ATTRIBUTES_RERANKER_INPUT_DOCUMENTS_COL,
    SPAN_ATTRIBUTES_RERANKER_OUTPUT_DOCUMENTS_COL,
    SPAN_ATTRIBUTES_RERANKER_QUERY_COL,
    SPAN_ATTRIBUTES_RERANKER_MODEL_NAME_COL,
    SPAN_ATTRIBUTES_RERANKER_TOP_K_COL,
    SPAN_ATTRIBUTES_SESSION_ID,
    SPAN_ATTRIBUTES_USER_ID,
    SPAN_ATTRIBUTES_METADATA,
    SPAN_ATTRIBUTES_LLM_TOOLS_COL,
]

# List of columns that must be present in the dataframe
SPAN_OPENINFERENCE_REQUIRED_COLUMNS = [
    col for col in SPAN_OPENINFERENCE_COLUMNS if col.required
]

# Eval columns
EVAL_COLUMN_PREFIX = "eval."
EVAL_LABEL_SUFFIX = ".label"
EVAL_SCORE_SUFFIX = ".score"
EVAL_EXPLANATION_SUFFIX = ".explanation"
# Eval column regex patterns
_EVAL_COLUMN_PREFIX_REGEX = r"eval\."
_EVAL_LABEL_SUFFIX_REGEX = r"\.label"
_EVAL_SCORE_SUFFIX_REGEX = r"\.score"
_EVAL_EXPLANATION_SUFFIX_REGEX = r"\.explanation"
_EVAL_NAME_REGEX = r"[a-zA-Z0-9_\s]+?"
# Eval column patterns
EVAL_COLUMN_PATTERN = (
    f"^{_EVAL_COLUMN_PREFIX_REGEX}{_EVAL_NAME_REGEX}"
    f"({_EVAL_LABEL_SUFFIX_REGEX}|{_EVAL_SCORE_SUFFIX_REGEX}|{_EVAL_EXPLANATION_SUFFIX_REGEX})$"
)
EVAL_NAME_PATTERN = rf"^{_EVAL_COLUMN_PREFIX_REGEX}({_EVAL_NAME_REGEX})\."
EVAL_LABEL_PATTERN = (
    f"^{_EVAL_COLUMN_PREFIX_REGEX}{_EVAL_NAME_REGEX}{_EVAL_LABEL_SUFFIX_REGEX}$"
)
EVAL_SCORE_PATTERN = (
    f"^{_EVAL_COLUMN_PREFIX_REGEX}{_EVAL_NAME_REGEX}{_EVAL_SCORE_SUFFIX_REGEX}$"
)
EVAL_EXPLANATION_PATTERN = f"^{_EVAL_COLUMN_PREFIX_REGEX}{_EVAL_NAME_REGEX}{_EVAL_EXPLANATION_SUFFIX_REGEX}$"
