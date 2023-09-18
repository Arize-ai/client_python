import json
from pathlib import Path

MAX_BYTES_PER_BULK_RECORD = 100000
MAX_DAYS_WITHIN_RANGE = 365
MIN_PREDICTION_ID_LEN = 1
MAX_PREDICTION_ID_LEN = 128
# The maximum number of character for tag values
MAX_TAG_LENGTH = 1000
# The maximum number of acceptable years in the past from current time for prediction_timestamps
MAX_PAST_YEARS_FROM_CURRENT_TIME = 2
# The maximum number of acceptable years in the future from current time for prediction_timestamps
MAX_FUTURE_YEARS_FROM_CURRENT_TIME = 1
# The maximum number of character for llm model name
MAX_LLM_MODEL_NAME_LENGTH = 50
# The maximum number of character for prompt template
MAX_PROMPT_TEMPLATE_LENGTH = 5000
# The maximum number of character for prompt template version
MAX_PROMPT_TEMPLATE_VERSION_LENGTH = 50
# The maximum number of embeddings
MAX_NUMBER_OF_EMBEDDINGS = 30

# Arize generated columns
GENERATED_PREDICTION_LABEL_COL = "arize_generated_prediction_label"
GENERATED_LLM_PARAMS_JSON_COL = "arize_generated_llm_params_json"

# Authentication via environment variables
SPACE_KEY_ENVVAR_NAME = "ARIZE_SPACE_KEY"
API_KEY_ENVVAR_NAME = "ARIZE_API_KEY"

path = Path(__file__).with_name("model_mapping.json")
with path.open("r") as f:
    MODEL_MAPPING_CONFIG = json.load(f)
