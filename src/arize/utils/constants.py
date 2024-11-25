import json
from pathlib import Path

MAX_BYTES_PER_BULK_RECORD = 100000
MAX_DAYS_WITHIN_RANGE = 365
MIN_PREDICTION_ID_LEN = 1
MAX_PREDICTION_ID_LEN = 512
MIN_DOCUMENT_ID_LEN = 1
MAX_DOCUMENT_ID_LEN = 128
# The maximum number of character for tag values
MAX_TAG_LENGTH = 20_000
MAX_TAG_LENGTH_TRUNCATION = 1_000
# The maximum number of character for embedding raw data
MAX_RAW_DATA_CHARACTERS = 2_000_000
MAX_RAW_DATA_CHARACTERS_TRUNCATION = 5_000
# The maximum number of acceptable years in the past from current time for prediction_timestamps
MAX_PAST_YEARS_FROM_CURRENT_TIME = 5
# The maximum number of acceptable years in the future from current time for prediction_timestamps
MAX_FUTURE_YEARS_FROM_CURRENT_TIME = 1
# The maximum number of character for llm model name
MAX_LLM_MODEL_NAME_LENGTH = 20_000
MAX_LLM_MODEL_NAME_LENGTH_TRUNCATION = 50
# The maximum number of character for prompt template
MAX_PROMPT_TEMPLATE_LENGTH = 50_000
MAX_PROMPT_TEMPLATE_LENGTH_TRUNCATION = 5_000
# The maximum number of character for prompt template version
MAX_PROMPT_TEMPLATE_VERSION_LENGTH = 20_000
MAX_PROMPT_TEMPLATE_VERSION_LENGTH_TRUNCATION = 50
# The maximum number of embeddings
MAX_NUMBER_OF_EMBEDDINGS = 30
MAX_EMBEDDING_DIMENSIONALITY = 20_000
# The maximum number of classes for multi class
MAX_NUMBER_OF_MULTI_CLASS_CLASSES = 300
MAX_MULTI_CLASS_NAME_LENGTH = 100
# The maximum number of references in embedding similarity search params
MAX_NUMBER_OF_SIMILARITY_REFERENCES = 10

# Arize generated columns
GENERATED_PREDICTION_LABEL_COL = "arize_generated_prediction_label"
GENERATED_LLM_PARAMS_JSON_COL = "arize_generated_llm_params_json"

# reserved columns for LLM run metadata
LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME = "total_token_count"
LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME = "prompt_token_count"
LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME = "response_token_count"
LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME = "response_latency_ms"

# all reserved tags
RESERVED_TAG_COLS = [
    LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
]


# Authentication via environment variables
SPACE_KEY_ENVVAR_NAME = "ARIZE_SPACE_KEY"
API_KEY_ENVVAR_NAME = "ARIZE_API_KEY"
DEVELOPER_KEY_ENVVAR_NAME = "ARIZE_DEVELOPER_KEY"
SPACE_ID_ENVVAR_NAME = "ARIZE_SPACE_ID"

# Default public Flight endpoint when not provided through env variable nor profile
DEFAULT_ARIZE_FLIGHT_HOST = "flight.arize.com"
DEFAULT_ARIZE_FLIGHT_PORT = 443

path = Path(__file__).with_name("model_mapping.json")
with path.open("r") as f:
    MODEL_MAPPING_CONFIG = json.load(f)
