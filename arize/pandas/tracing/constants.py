# The defualt format used to parse datetime objects from strings
DEFAULT_DATETIME_FMT = "%Y-%m-%dT%H:%M:%S.%f+00:00"
# Minumum/Maximum number of characters for span/trace/parent ids in spans
SPAN_ID_MIN_STR_LENGTH = 12
SPAN_ID_MAX_STR_LENGTH = 128
# Minumum/Maximum number of characters for span name
SPAN_NAME_MIN_STR_LENGTH = 0
SPAN_NAME_MAX_STR_LENGTH = 50
# Minumum/Maximum number of characters for span status message
SPAN_STATUS_MSG_MIN_STR_LENGTH = 0
SPAN_STATUS_MSG_MAX_STR_LENGTH = 10_000
# Minumum/Maximum number of characters for span event name
SPAN_EVENT_NAME_MAX_STR_LENGTH = 100
# Minumum/Maximum number of characters for span event attributes
SPAN_EVENT_ATTRS_MAX_STR_LENGTH = 10_000
# Maximum number of characters for span kind
SPAN_KIND_MAX_STR_LENGTH = 100
SPAN_EXCEPTION_TYPE_MAX_STR_LENGTH = 100
SPAN_EXCEPTION_MESSAGE_MAX_STR_LENGTH = 100
SPAN_EXCEPTION_STACK_TRACE_MAX_STR_LENGTH = 10_000
SPAN_IO_VALUE_MAX_STR_LENGTH = 4_000_000
SPAN_IO_MIME_TYPE_MAX_STR_LENGTH = 100
SPAN_EMBEDDING_NAME_MAX_STR_LENGTH = 100
SPAN_EMBEDDING_TEXT_MAX_STR_LENGTH = 4_000_000
SPAN_LLM_MODEL_NAME_MAX_STR_LENGTH = 100
SPAN_LLM_MESSAGE_ROLE_MAX_STR_LENGTH = 100
SPAN_LLM_MESSAGE_CONTENT_MAX_STR_LENGTH = 4_000_000
SPAN_LLM_TOOL_CALL_FUNCTION_NAME_MAX_STR_LENGTH = 500
SPAN_LLM_PROMPT_TEMPLATE_MAX_STR_LENGTH = 4_000_000
SPAN_LLM_PROMPT_TEMPLATE_VARIABLES_MAX_STR_LENGTH = 10_000
SPAN_LLM_PROMPT_TEMPLATE_VERSION_MAX_STR_LENGTH = 100
SPAN_TOOL_NAME_MAX_STR_LENGTH = 100
SPAN_TOOL_DESCRIPTION_MAX_STR_LENGTH = 1_000
SPAN_TOOL_PARAMETERS_MAX_STR_LENGTH = 1_000
SPAN_RERANKER_QUERY_MAX_STR_LENGTH = 10_000
SPAN_RERANKER_MODEL_NAME_MAX_STR_LENGTH = 100
SPAN_DOCUMENT_ID_MAX_STR_LENGTH = 100
SPAN_DOCUMENT_CONTENT_MAX_STR_LENGTH = 4_000_000
JSON_STRING_MAX_STR_LENGTH = 4_000_000
# Eval related constants
EVAL_LABEL_MIN_STR_LENGTH = 1  # we do not accept empty strings
EVAL_LABEL_MAX_STR_LENGTH = 100
EVAL_EXPLANATION_MAX_STR_LENGTH = 10_000
