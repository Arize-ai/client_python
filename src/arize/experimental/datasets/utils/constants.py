from enum import Enum

from openinference.semconv import trace

from arize.pandas.proto import requests_pb2 as pb2

INFERENCES = pb2.INFERENCES
GENERATIVE = pb2.GENERATIVE

"""Internal Use"""

# Default API endpoint when not provided through env variable nor profile
DEFAULT_ARIZE_FLIGHT_HOST = "flight.arize.com"
DEFAULT_ARIZE_FLIGHT_PORT = 443
DEFAULT_ARIZE_OTLP_ENDPOINT = "https://otlp.arize.com/v1"

# Name of the current package.
DEFAULT_PACKAGE_NAME = "arize_python_datasets_client"

# Default headers to trace and help identify requests. For debugging.
DEFAULT_ARIZE_SESSION_ID = "x-arize-session-id"  # Generally the session name.
DEFAULT_ARIZE_TRACE_ID = "x-arize-trace-id"
DEFAULT_PACKAGE_VERSION = "x-package-version"

# Default initial wait time for retries in seconds.
DEFAULT_RETRY_INITIAL_WAIT_TIME = 0.25

# Default maximum wait time for retries in seconds.
DEFAULT_RETRY_MAX_WAIT_TIME = 10.0

# Default to use grpc + tls scheme.
DEFAULT_TRANSPORT_SCHEME = "grpc+tls"


class FlightActionKey(Enum):
    GET_DATASET_VERSION = "get_dataset_version"
    LIST_DATASETS = "list_datasets"
    DELETE_DATASET = "delete_dataset"
    CREATE_EXPERIMENT_DB_ENTRY = "create_experiment_db_entry"


OPEN_INFERENCE_JSON_STR_TYPES = frozenset(
    [
        trace.DocumentAttributes.DOCUMENT_METADATA,
        trace.SpanAttributes.LLM_FUNCTION_CALL,
        trace.SpanAttributes.LLM_INVOCATION_PARAMETERS,
        trace.SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
        trace.MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON,
        trace.SpanAttributes.METADATA,
        trace.SpanAttributes.TOOL_PARAMETERS,
        trace.ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
    ]
)
