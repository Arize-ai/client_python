"""Span data conversion utilities for transforming and normalizing span data."""

import json
from collections.abc import Iterable
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from arize.spans.columns import SPAN_OPENINFERENCE_COLUMNS, SpanColumnDataType


def convert_timestamps(df: pd.DataFrame, fmt: str = "") -> pd.DataFrame:
    """Convert timestamp columns in a :class:`pandas.DataFrame` to nanoseconds.

    Args:
        df: The :class:`pandas.DataFrame` containing timestamp columns.
        fmt: Optional datetime format string for parsing string timestamps. Defaults to "".

    Returns:
        The :class:`pandas.DataFrame` with timestamp columns converted to nanoseconds.

    Raises:
        KeyError: If required timestamp column is not found in :class:`pandas.DataFrame`.
    """
    for col in SPAN_OPENINFERENCE_COLUMNS:
        if col.data_type != SpanColumnDataType.TIMESTAMP:
            continue
        if col.name not in df.columns:
            raise KeyError(f"Column '{col.name}' not found in DataFrame")
        df[col.name] = df[col.name].apply(lambda dt: _datetime_to_ns(dt, fmt))
    return df


def _datetime_to_ns(dt: object, fmt: str) -> int:
    if isinstance(dt, str):
        # Try ISO 8601 with timezone first
        try:
            parsed = datetime.fromisoformat(dt)
            if parsed.tzinfo is None:
                # If no timezone, assume UTC
                parsed = parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            # Fall back to custom format
            parsed = datetime.strptime(dt, fmt).replace(tzinfo=timezone.utc)

        return int(parsed.timestamp() * 1e9)
    if isinstance(dt, datetime):
        return int(datetime.timestamp(dt) * 1e9)
    if isinstance(dt, pd.Timestamp):
        return int(dt.value)
    if isinstance(dt, pd.DatetimeIndex):
        # Only allow a single element; otherwise ambiguous for a scalar function
        if len(dt) != 1:
            raise TypeError(
                f"Expected a single timestamp in DatetimeIndex, got length={len(dt)}"
            )
        return int(dt.to_numpy(dtype="datetime64[ns]").astype("int64")[0])
    if isinstance(dt, (int, float)):
        # Assume value already in nanoseconds,
        # validate timestamps in validate_values
        return int(dt)
    e = TypeError(f"Cannot convert type {type(dt)} to nanoseconds")
    # logger.error(f"Error converting pandas Timestamp to nanoseconds: {e}")
    raise e


def jsonify_dictionaries(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dictionary and list-of-dictionary columns to JSON strings.

    Args:
        df: The :class:`pandas.DataFrame` containing dictionary columns.

    Returns:
        The DataFrame with dictionary columns converted to JSON strings.
    """
    # NOTE: numpy arrays are not json serializable. Hence, we assume the
    # embeddings come as lists, not arrays
    dict_cols = [
        col
        for col in SPAN_OPENINFERENCE_COLUMNS
        if col.data_type == SpanColumnDataType.DICT
    ]
    list_of_dict_cols = [
        col
        for col in SPAN_OPENINFERENCE_COLUMNS
        if col.data_type == SpanColumnDataType.LIST_DICT
    ]
    for col in dict_cols:
        col_name = col.name
        if col_name not in df.columns:
            # logger.debug(f"passing on {col_name}")
            continue
        # logger.debug(f"jsonifying {col_name}")
        df[col_name] = df[col_name].apply(lambda d: _jsonify_dict(d))

    for col in list_of_dict_cols:
        col_name = col.name
        if col_name not in df.columns:
            # logger.debug(f"passing on {col_name}")
            continue
        # logger.debug(f"jsonifying {col_name}")
        df[col_name] = df[col_name].apply(
            lambda list_of_dicts: _jsonify_list_of_dicts(list_of_dicts)
        )
    return df


# Defines what is considered a missing value
def is_missing_value(value: object) -> bool:
    """Check if a value should be considered missing or invalid.

    Args:
        value: The value to check.

    Returns:
        True if the value is missing (NaN, infinity, or pandas NA), False otherwise.
    """
    assumed_missing_values = (
        np.inf,
        -np.inf,
    )
    return value in assumed_missing_values or pd.isna(value)  # type: ignore[call-overload]


def _jsonify_list_of_dicts(
    list_of_dicts: Iterable[dict[str, object]] | None,
) -> list[str]:
    if list_of_dicts is None or is_missing_value(list_of_dicts):
        return []
    return [
        result
        for d in list_of_dicts
        if (result := _jsonify_dict(d)) is not None
    ]


def _jsonify_dict(d: dict[str, object] | None) -> str | None:
    if d is None:
        return None
    if is_missing_value(d):
        return None
    d = d.copy()  # avoid side effects
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        if isinstance(v, dict):
            d[k] = _jsonify_dict(v)
    return json.dumps(d, ensure_ascii=False)
