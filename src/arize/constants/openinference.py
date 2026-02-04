"""OpenInference semantic convention constants and attribute definitions."""

import openinference.semconv.trace as oinf

OPEN_INFERENCE_JSON_STR_TYPES = frozenset(
    [
        oinf.DocumentAttributes.DOCUMENT_METADATA,
        oinf.SpanAttributes.LLM_FUNCTION_CALL,
        oinf.SpanAttributes.LLM_INVOCATION_PARAMETERS,
        oinf.SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
        oinf.MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON,
        oinf.SpanAttributes.METADATA,
        oinf.SpanAttributes.TOOL_PARAMETERS,
        oinf.ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
    ]
)
