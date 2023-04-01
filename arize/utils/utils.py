import base64
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.wrappers_pb2 import StringValue

from .. import public_pb2 as pb2
from .types import Embedding

MAX_BYTES_PER_BULK_RECORD = 100000
MAX_DAYS_WITHIN_RANGE = 365
MIN_PREDICTION_ID_LEN = 1
MAX_PREDICTION_ID_LEN = 128
MODEL_MAPPING_CONFIG = None

path = Path(__file__).with_name("model_mapping.json")
with path.open("r") as f:
    MODEL_MAPPING_CONFIG = json.load(f)


def validate_prediction_timestamps(prediction_ids, prediction_timestamps):
    if prediction_timestamps is None:
        return
    else:
        expected_count = prediction_ids.shape[0]

    if isinstance(prediction_timestamps, pd.Series):
        if prediction_timestamps.shape[0] != expected_count:
            raise ValueError(
                f"prediction_timestamps has {prediction_timestamps.shape[0]} elements, but must "
                f"have same number of "
                f"elements as prediction_ids: {expected_count}. "
            )
    elif isinstance(prediction_timestamps, list):
        if len(prediction_timestamps) != expected_count:
            raise ValueError(
                f"prediction_timestamps has length {len(prediction_timestamps)} but must have "
                f"same number of elements as "
                f"prediction_ids: {expected_count}. "
            )
    else:
        raise TypeError(
            f"prediction_timestamps is type {type(prediction_timestamps)}, but expected one of: "
            f"pd.Series, list<int>"
        )
    now = int(time.time())
    for ts in prediction_timestamps:
        if not is_timestamp_in_range(now, ts):
            raise ValueError(
                f"timestamp: {ts} in prediction_timestamps is out of range. Value must be within "
                f"1 year of the current time."
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


def get_bulk_records(space_key: str, model_id: str, model_version: Optional[str], records):
    for k, r in records.items():
        records[k] = pb2.BulkRecord(
            records=r,
            space_key=space_key,
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


def convert_dictionary(d):
    # Takes a dictionary and
    # - casts the keys as strings
    # - turns the values of the dictionary to our proto values pb2.Value()
    converted_dict = {}
    for (k, v) in d.items():
        val = get_value_object(value=v, name=k)
        if val is not None:
            converted_dict[str(k)] = val
    return converted_dict


def get_value_object(name: Union[str, int, float], value):
    if isinstance(value, pb2.Value):
        return value
    # The following `convert_element` done in single log validation
    # of features & tags. It is not done in bulk_log
    val = convert_element(value)
    if val is None:
        return None
    if isinstance(val, (str, bool)):
        return pb2.Value(string=str(val))
    if isinstance(val, int):
        return pb2.Value(int=val)
    if isinstance(val, float):
        return pb2.Value(double=val)
    if isinstance(val, Embedding):
        return pb2.Value(embedding=get_value_embedding(val))
    else:
        raise TypeError(
            f'dimension "{name}" = {value} is type {type(value)}, but must be one of: bool, str, '
            f"float, int, embedding"
        )


def get_value_embedding(val: Embedding) -> pb2.Embedding:
    if Embedding._is_valid_iterable(val.data):
        return pb2.Embedding(
            vector=val.vector,
            raw_data=pb2.Embedding.RawData(tokenArray=pb2.Embedding.TokenArray(tokens=val.data)),
            link_to_data=StringValue(value=val.link_to_data),
        )
    elif isinstance(val.data, str):
        return pb2.Embedding(
            vector=val.vector,
            raw_data=pb2.Embedding.RawData(
                tokenArray=pb2.Embedding.TokenArray(tokens=[val.data])
                # Convert to list of 1 string
            ),
            link_to_data=StringValue(value=val.link_to_data),
        )
    elif val.data is None:
        return pb2.Embedding(
            vector=val.vector,
            link_to_data=StringValue(value=val.link_to_data),
        )

    return None


def get_timestamp(time_overwrite):
    if time_overwrite is None:
        return None
    time = convert_element(time_overwrite)
    if not isinstance(time_overwrite, int):
        raise TypeError(
            f"time_overwrite {time_overwrite} is type {type(time_overwrite)}, but expects int. ("
            f"Unix epoch "
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


def reconstruct_url(response: Any):
    returnedUrl = json.loads(response.content.decode())["realTimeIngestionUri"]
    parts = returnedUrl.split("/")
    encodedOrg = base64.b64encode(f"AccountOrganization:{parts[4]}".encode()).decode()
    encodedSpace = base64.b64encode(f"Space:{parts[6]}".encode()).decode()
    reconstructed = (
        f"https://{parts[2]}/organizations/{encodedOrg}/spaces/"
        f"{encodedSpace}/models/modelName/{parts[-1]}"
    )
    return reconstructed


def get_python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
