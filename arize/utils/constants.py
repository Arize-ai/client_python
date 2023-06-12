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

path = Path(__file__).with_name("model_mapping.json")
with path.open("r") as f:
    MODEL_MAPPING_CONFIG = json.load(f)
