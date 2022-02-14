import math
import time
import pandas as pd

from typing import Union, Optional, Tuple

from arize import public_pb2 as public__pb2
from arize.utils.types import ModelTypes
from google.protobuf.timestamp_pb2 import Timestamp

MAX_BYTES_PER_BULK_RECORD = 100000
MAX_DAYS_WITHIN_RANGE = 365


def infer_model_type(
    label: Union[str, bool, int, float, Tuple[str, float]]
) -> ModelTypes:
    val = convert_element(label)
    if isinstance(val, bool):
        return ModelTypes.BINARY
    elif isinstance(val, (int, float)):
        return ModelTypes.NUMERIC
    elif isinstance(val, str):
        return ModelTypes.CATEGORICAL
    elif isinstance(val, tuple):
        return ModelTypes.SCORE_CATEGORICAL
    raise TypeError(
        f"label {label} has type {type(label)}, but must be one of str, bool, float, or int"
    )


def validate_prediction_timestamps(prediction_ids, prediction_timestamps):
    if prediction_timestamps is None:
        return
    else:
        expected_count = prediction_ids.shape[0]

    if isinstance(prediction_timestamps, pd.Series):
        if prediction_timestamps.shape[0] != expected_count:
            raise ValueError(
                f"prediction_timestamps has {prediction_timestamps.shape[0]} elements, but must have same number of "
                f"elements as prediction_ids: {expected_count}. "
            )
    elif isinstance(prediction_timestamps, list):
        if len(prediction_timestamps) != expected_count:
            raise ValueError(
                f"prediction_timestamps has length {len(prediction_timestamps)} but must have same number of elements as "
                f"prediction_ids: {expected_count}. "
            )
    else:
        raise TypeError(
            f"prediction_timestamps is type {type(prediction_timestamps)}, but expected one of: pd.Series, list<int>"
        )
    now = int(time.time())
    for ts in prediction_timestamps:
        if not is_timestamp_in_range(now, ts):
            raise ValueError(
                f"timestamp: {ts} in prediction_timestamps is out of range. Value must be within 1 year of the current time."
            )


def num_chunks(records):
    total_bytes = sum(r.ByteSize() for r in records)
    num_of_bulk = math.ceil(total_bytes / MAX_BYTES_PER_BULK_RECORD)
    return math.ceil(len(records) / num_of_bulk)


def bundle_records(records) -> {}:
    recs_per_msg = num_chunks(records)
    recs = {
        (i, i + recs_per_msg): records[i : i + recs_per_msg]
        for i in range(0, len(records), recs_per_msg)
    }
    return recs


def get_bulk_records(
    organization_key: str, model_id: str, model_version: Optional[str], records
):
    for k, r in records.items():
        records[k] = public__pb2.BulkRecord(
            records=r,
            organization_key=organization_key,
            model_id=model_id,
            model_version=model_version,
        )
    return records


def convert_element(value):
    """converts scalar or array to python native"""
    val = getattr(value, "tolist", lambda: value)()
    # Check if it's a list since elements from pd indices are converted to a scalar
    # whereas pd series/dataframe elements are converted to list of 1 with the native value
    if isinstance(val, list):
        val = val[0] if val else None
    if pd.isna(val):
        return None
    return val


def get_value_object(name: Union[str, int, float], value):
    if isinstance(value, public__pb2.Value):
        return value
    val = convert_element(value)
    if val is None:
        return None
    if isinstance(val, (str, bool)):
        return public__pb2.Value(string=str(val))
    if isinstance(val, int):
        return public__pb2.Value(int=val)
    if isinstance(val, float):
        return public__pb2.Value(double=val)
    else:
        raise TypeError(
            f'dimension "{name}" = {value} is type {type(value)}, but must be one of: bool, str, float, int.'
        )


def get_timestamp(time_overwrite):
    if time_overwrite is None:
        return None
    time = convert_element(time_overwrite)
    if not isinstance(time_overwrite, int):
        raise TypeError(
            f"time_overwrite {time_overwrite} is type {type(time_overwrite)}, but expects int. (Unix epoch "
            f"time in seconds) "
        )
    ts = Timestamp()
    ts.FromSeconds(time)
    return ts


def is_timestamp_in_range(now: int, ts: int):
    max_time = now + (MAX_DAYS_WITHIN_RANGE * 24 * 60 * 60)
    min_time = now - (MAX_DAYS_WITHIN_RANGE * 24 * 60 * 60)
    if ts > max_time:
        return False
    if ts < min_time:
        return False
    return True
