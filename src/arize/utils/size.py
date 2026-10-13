"""Size calculation utilities for payloads and data structures."""

import json
from collections.abc import Iterator
from typing import Any

import pandas as pd

from arize.utils.records import normalize_json_value

_BYTES_PER_MB = 1024 * 1024
# The JSON encoder writes "[" and "]" around a list and ", " between items.
_LIST_BRACKETS_BYTES = 2
_ITEM_SEPARATOR_BYTES = 2


def _json_default(value: object) -> object:
    normalized = normalize_json_value(value)
    return str(value) if normalized is value else normalized


def _iter_rows(
    payload: list[dict[str, Any]] | pd.DataFrame,
) -> Iterator[dict[Any, Any]]:
    if isinstance(payload, pd.DataFrame):
        columns = list(payload.columns)
        for values in payload.itertuples(index=False, name=None):
            yield dict(zip(columns, values, strict=True))
    elif isinstance(payload, list):
        yield from payload
    else:
        raise TypeError(f"Unsupported payload type: {type(payload)}")


def to_json_records_within(
    payload: list[dict[str, Any]] | pd.DataFrame, max_mb: float
) -> list[dict[str, Any]] | None:
    """Convert ``payload`` to JSON-compatible records if its JSON body fits.

    Rows are converted one at a time while the size of the serialized request
    body is added up. Returns ``None`` as soon as the body would exceed
    ``max_mb``, without converting the remaining rows.
    """
    max_bytes = max_mb * _BYTES_PER_MB
    total_bytes = _LIST_BRACKETS_BYTES
    records: list[dict[str, Any]] = []
    for row in _iter_rows(payload):
        record = normalize_json_value(row)
        if records:
            total_bytes += _ITEM_SEPARATOR_BYTES
        total_bytes += len(
            json.dumps(record, default=_json_default).encode("utf-8")
        )
        if total_bytes > max_bytes:
            return None
        records.append(record)  # type: ignore[arg-type]
    return records
