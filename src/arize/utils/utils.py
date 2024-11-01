# type: ignore[pb2]
import json
import math
import sys
from typing import Any, Optional, Union

import pandas as pd
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.wrappers_pb2 import StringValue

from arize.utils.constants import (
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
)

from .. import public_pb2 as pb2
from .constants import MAX_BYTES_PER_BULK_RECORD
from .types import Embedding, Schema, _PromptOrResponseText, is_list_of


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
    space_key: str, model_id: str, model_version: Optional[str], records
):
    for k, r in records.items():
        records[k] = pb2.BulkRecord(
            records=r,
            space_key=space_key,
            model_id=model_id,
            model_version=model_version,
        )
    return records


def convert_element(value):
    """Converts scalar or array to python native"""
    val = getattr(value, "tolist", lambda: value)()
    # Check if it's a list since elements from pd indices are converted to a
    # scalar whereas pd series/dataframe elements are converted to list of 1
    # with the native value
    if isinstance(val, list):
        val = val[0] if val else None
    if pd.isna(val):
        return None
    return val


def convert_dictionary(d):
    if d is None:
        return {}
    # Takes a dictionary and
    # - casts the keys as strings
    # - turns the values of the dictionary to our proto values pb2.Value()
    converted_dict = {}
    for k, v in d.items():
        val = get_value_object(value=v, name=k)
        if val is not None:
            converted_dict[str(k)] = val
    return converted_dict


def get_value_object(name: Union[str, int, float], value):
    if isinstance(value, pb2.Value):
        return value
    if value is not None and is_list_of(value, str):
        return pb2.Value(multi_value=pb2.MultiValue(values=value))
    # The following `convert_element` done in single log validation
    # of features & tags. It is not done in bulk_log
    val = convert_element(value)
    if val is None:
        return None
    elif isinstance(val, (str, bool)):
        return pb2.Value(string=str(val))
    elif isinstance(val, int):
        return pb2.Value(int=val)
    elif isinstance(val, float):
        return pb2.Value(double=val)
    elif isinstance(val, Embedding):
        return pb2.Value(embedding=get_proto_embedding(val))
    elif isinstance(val, _PromptOrResponseText):
        return pb2.Value(
            embedding=get_proto_embedding_from_prompt_or_response_test(val)
        )
    else:
        raise TypeError(
            f"dimension '{name}' = {value} is type {type(value)}, but must be "
            "one of: bool, str, float, int, embedding"
        )


def get_proto_embedding(val: Embedding) -> pb2.Embedding:
    if Embedding._is_valid_iterable(val.data):
        return pb2.Embedding(
            vector=val.vector,
            raw_data=pb2.Embedding.RawData(
                tokenArray=pb2.Embedding.TokenArray(tokens=val.data)
            ),
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


def get_proto_embedding_from_prompt_or_response_test(
    val: _PromptOrResponseText,
) -> pb2.Embedding:
    return pb2.Embedding(
        raw_data=pb2.Embedding.RawData(
            tokenArray=pb2.Embedding.TokenArray(tokens=[val.data])
            # Convert to list of 1 string
        ),
    )


def get_timestamp(time_overwrite):
    if time_overwrite is None:
        return None
    time = convert_element(time_overwrite)
    if not isinstance(time_overwrite, int):
        raise TypeError(
            f"time_overwrite {time_overwrite} is type {type(time_overwrite)}, "
            "but expects int (Unix epoch time in seconds)."
        )
    ts = Timestamp()
    ts.FromSeconds(time)
    return ts


def is_timestamp_in_range(now: int, ts: int):
    max_time = now + (MAX_FUTURE_YEARS_FROM_CURRENT_TIME * 365 * 24 * 60 * 60)
    min_time = now - (MAX_PAST_YEARS_FROM_CURRENT_TIME * 365 * 24 * 60 * 60)
    return min_time <= ts <= max_time


def reconstruct_url(response: Any, drop_in_data_ingestion: bool = True):
    if "realTimeIngestionUri" in json.loads(response.content.decode()):
        return json.loads(response.content.decode())["realTimeIngestionUri"]
    return ""


def get_python_version():
    return (
        f"{sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )


def is_delayed_schema(schema: Schema) -> bool:
    """
    This function checks if the given schema, according to the columns provided
    by the user, has inherently latent information
    Args:
        schema (Schema): The schema to analyze

    Returns:
        bool: True if the schema is "delayed", i.e., does not possess prediction
        columns and has actual or feature importance columns.
    """
    return (
        schema.has_actual_columns() or schema.has_feature_importance_columns()
    ) and not schema.has_prediction_columns()


def is_python_version_below_required_min(min_req_version: str) -> None:
    min_major = int(min_req_version.split(".")[0])
    min_minor = int(min_req_version.split(".")[1])
    min_micro = int(min_req_version.split(".")[2])
    min_version = (min_major, min_minor, min_micro)
    version = sys.version_info[:3]

    return version < min_version


# Resets the dataframe index if it is not a RangeIndex
def reset_dataframe_index(dataframe: pd.DataFrame) -> None:
    if not isinstance(dataframe.index, pd.RangeIndex):
        drop = dataframe.index.name in dataframe.columns
        dataframe.reset_index(inplace=True, drop=drop)
