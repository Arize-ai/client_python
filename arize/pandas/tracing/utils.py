import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from arize.pandas.tracing.constants import ASSUMED_MISSING_VALUES
from arize.utils.logging import logger

from .columns import SPAN_OPENINFERENCE_COLUMNS, SpanColumnDataType


def convert_timestamps(df: pd.DataFrame, fmt: str = "") -> pd.DataFrame:
    time_cols = [
        col for col in SPAN_OPENINFERENCE_COLUMNS if col.data_type == SpanColumnDataType.TIMESTAMP
    ]
    for col in time_cols:
        df[col.name] = df[col.name].apply(lambda dt: _datetime_to_ns(dt, fmt))
    return df


def _datetime_to_ns(dt: Union[str, datetime], fmt: str) -> int:
    if isinstance(dt, str):
        try:
            ts = int(datetime.timestamp(datetime.strptime(dt, fmt)) * 1e9)
        except Exception as e:
            logger.error(
                f"Error parsing string '{dt}' to timestamp in nanoseconds "
                f"using the format '{fmt}': {e}"
            )
            raise e
        return ts
    elif isinstance(dt, datetime):
        try:
            ts = int(datetime.timestamp(dt) * 1e9)
        except Exception as e:
            logger.error(f"Error converting datetime object to nanoseconds: {e}")
            raise e
        return ts
    elif isinstance(dt, pd.Timestamp) or isinstance(pd.DatetimeIndex):
        try:
            ts = int(datetime.timestamp(dt.to_pydatetime()) * 1e9)
        except Exception as e:
            logger.error(f"Error converting pandas Timestamp to nanoseconds: {e}")
            raise e
        return ts
    elif isinstance(dt, (int, float)):
        # Assume value already in nanoseconds,
        # validate timestamps in validate_values
        return dt
    else:
        e = TypeError(f"Cannot convert type {type(dt)} to nanoseconds")
        logger.error(f"Error converting pandas Timestamp to nanoseconds: {e}")
        raise e


def jsonify_dictionaries(df: pd.DataFrame) -> pd.DataFrame:
    # NOTE: numpy arrays are not json serializable. Hence, we assume the
    # embeddings come as lists, not arrays
    dict_cols = [
        col for col in SPAN_OPENINFERENCE_COLUMNS if col.data_type == SpanColumnDataType.DICT
    ]
    list_of_dict_cols = [
        col for col in SPAN_OPENINFERENCE_COLUMNS if col.data_type == SpanColumnDataType.LIST_DICT
    ]
    for col in dict_cols:
        col_name = col.name
        if col_name not in df.columns:
            logger.debug(f"passing on {col_name}")
            continue
        logger.debug(f"jsonifying {col_name}")
        df[col_name] = df[col_name].apply(lambda d: _jsonify_dict(d))

    for col in list_of_dict_cols:
        col_name = col.name
        if col_name not in df.columns:
            logger.debug(f"passing on {col_name}")
            continue
        logger.debug(f"jsonifying {col_name}")
        df[col_name] = df[col_name].apply(
            lambda list_of_dicts: _jsonify_list_of_dicts(list_of_dicts)
        )
    return df


def _jsonify_list_of_dicts(
    list_of_dicts: Optional[Iterable[Dict[str, Any]]],
) -> Optional[List[str]]:
    if not isinstance(list_of_dicts, Iterable) and list_of_dicts in ASSUMED_MISSING_VALUES:
        return None
    list_of_json = []
    for d in list_of_dicts:
        list_of_json.append(_jsonify_dict(d))
    return list_of_json


def _jsonify_dict(d: Optional[Dict[str, Any]]) -> Optional[str]:
    if d in ASSUMED_MISSING_VALUES:
        return None
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
    return json.dumps(d, ensure_ascii=False)
