"""Size calculation utilities for payloads and data structures."""

import sys
from typing import Any

import pandas as pd


def get_payload_size_mb(payload: list[dict[str, Any]] | pd.DataFrame) -> float:
    """Return approximate size of payload in MB."""
    if isinstance(payload, pd.DataFrame):
        # memory_usage(deep=True) sums all columns' memory footprint
        size_bytes = payload.memory_usage(deep=True).sum()
    elif isinstance(payload, list):
        # sys.getsizeof() gives shallow size; sum all element sizes for rough total
        size_bytes = sys.getsizeof(payload) + sum(
            sys.getsizeof(x) for x in payload
        )
    else:
        raise TypeError(f"Unsupported payload type: {type(payload)}")

    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 3)
