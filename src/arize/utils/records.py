"""Conversion of row inputs into JSON-compatible records."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd


def normalize_json_value(value: object) -> object:
    """Convert NumPy containers and scalars to JSON-compatible Python values.

    NaN and Infinity become None: `json.dumps` writes them as bare `NaN`/
    `Infinity` tokens, which are not valid JSON and the server rejects.
    """
    if value is pd.NA:
        return None
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else np.datetime_as_string(value)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.datetime64):
            normalized = np.datetime_as_string(value).astype(object)
            normalized[np.isnat(value)] = None
            return normalized.tolist()
        return normalize_json_value(value.tolist())
    if isinstance(value, np.generic):
        return normalize_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    return value


def to_json_records(
    data: list[dict[str, object]] | pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert a DataFrame or list of dicts to records holding only JSON-compatible values."""
    records = (
        data.to_dict(orient="records")
        if isinstance(data, pd.DataFrame)
        else data
    )
    return cast("list[dict[str, Any]]", normalize_json_value(records))
