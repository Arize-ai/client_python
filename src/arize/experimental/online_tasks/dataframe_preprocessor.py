import json
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from arize.utils.logging import logger


def extract_nested_data_to_column(
    attributes: List[str], df: pd.DataFrame
) -> pd.DataFrame:
    """
    This function, used in Online Tasks, is typically run on data exported from Arize.
    It prepares the DataFrame by extracting relevant attributes from complex, deeply nested
    data structures, such as those found in LLM outputs or JSON-like records. It helps extract
    specific values from these nested structures by identifying the longest matching column name
    in the DataFrame and recursively accessing the desired attribute path within each row.
    This preprocessing step ensures that the extracted values are available as new columns,
    allowing evaluators to process and assess these values effectively.

    For each attributes string in `attributes` (e.g. "attributes.llm.output_messages.0.message.content"),
    1) Find the largest prefix that is actually a column name in `df`. (e.g. "attributes.llm.output_messages")
    2) Use the remainder of the attribute as the introspect path for the values in that column:
        - Calls _introspect_arize_attribute({row_value}, {attribute_remainder}) for each row value
            e.g. {row_value} = [{'message.role': 'assistant',
                                'message.content': 'The capital of China is Beijing.'}]
            e.g. {attribute_remainder} = "0.message.content"
        - This introspect function recursively indexes into a given row_value based on
            the attribute_remainder path and is able to handle a variety of nested structures
            such as the example given for {row_value}
    3) Create a new column named exactly `attribute`, filling it row-by-row with the result
       of introspecting into the column's value. (e.g. row extracted: 'The capital of China is Beijing.')
       If introspect fails or yields None, store NaN.
    4) After all columns have been created, drop rows that have NaN in *any* of the newly-created columns.
    5) Log how many rows were dropped and, if zero rows remain, log a message indicating that
       there are no rows satisfying *all* of the queries.
    """

    # Make a copy so as not to alter the input df
    result_df = df.copy()

    # Keep track of which new columns we add. Each column name will match each user-inputted attribute
    # (e.g. "attributes.llm.output_messages.0.message.content")
    new_cols: List[str] = []

    for attribute in attributes:
        parts = attribute.split(".")
        prefix_col = None
        prefix_len = 0

        # 1) Find largest prefix of attribute that matches a column in df
        for i in range(1, len(parts) + 1):
            candidate = ".".join(parts[:i])
            if candidate in result_df.columns:
                prefix_col = candidate
                prefix_len = i

        if prefix_col is None:
            raise Exception("No such column found in DataFrame.")

        # 2) The remainder after the prefix
        remainder = ".".join(parts[prefix_len:])

        # 3) Apply introspect row-by-row
        def apply_introspect_arize_attribute(
            row: pd.Series,
            prefix_col: str = prefix_col,
            remainder: str = remainder,
        ) -> Any:
            val = row[prefix_col]
            try:
                result = _introspect_arize_attribute(val, remainder)
                return result if result is not None else np.nan
            except Exception:
                return np.nan

        result_df[attribute] = result_df.apply(
            apply_introspect_arize_attribute, axis=1
        )

        new_cols.append(attribute)

    # 4) Drop rows that are NaN in *any* of the newly-added columns
    rows_before = len(df)
    result_df = result_df.dropna(subset=new_cols)
    rows_after = len(result_df)
    rows_dropped = rows_before - rows_after

    # 5) Log some diagnostics
    logger.info(f"Rows before processing: {rows_before}")
    logger.info(f"Rows after processing: {rows_after}")
    logger.info(f"Rows dropped: {rows_dropped}")

    if rows_after == 0:
        logger.info(
            f"For the given filter, there are no rows that have ALL of the following variables: {attributes}"
        )

    return result_df


def _introspect_arize_attribute(value: Any, attribute: str) -> Any:
    """
    Recursively drill into `value` following the dot-delimited `attribute`.
    Example:
        value: [{'message.role': 'assistant', 'message.content': 'The capital of China is Beijing.'}]
        attribute: "0.message.content"
        Returns: 'The capital of China is Beijing.'

      - Returns None immediately when a key or index is not found
      - Handles integer parts for lists
      - Parses JSON strings
      - Converts NumPy arrays to lists
      - Allows dotted keys (e.g. "message.content") by combining parts

    """
    if not attribute:
        return value

    attribute_parts = attribute.split(".")
    return _introspect_arize_attribute_parts(value, attribute_parts)


def _introspect_arize_attribute_parts(
    current_value: Any, attribute_parts_unprocessed: List[str]
) -> Any:
    # If no more parts, we return whatever we have
    if not attribute_parts_unprocessed:
        return current_value

    current_value = _ensure_deserialized(current_value)

    # Parse out the next value using the first (or combined) part(s).
    parsed_value, num_parts_processed = _parse_value(
        current_value, attribute_parts_unprocessed
    )

    # If we can't find a match, immediately return None
    if parsed_value is None:
        return None

    # Otherwise, recurse deeper with the leftover parts
    return _introspect_arize_attribute_parts(
        parsed_value, attribute_parts_unprocessed[num_parts_processed:]
    )


def _parse_value(
    current_value: Any, attribute_parts_unprocessed: List[str]
) -> Tuple[Optional[Any], int]:
    """
    Attempt to parse out the next value from `current_value` using the earliest parts:

    1) If `attribute_parts_unprocessed[0]` is an integer index and `current_value` is a list/tuple,
       index into it.
    2) Else if `current_value` is a dict, check if `attribute_parts_unprocessed[0]` is a key.
       If not found, try combining `attribute_parts_unprocessed[0] + '.' + attribute_parts_unprocessed[1]`...
       to handle dotted keys in the dict.
    3) If none match, return (None, 1) to signal "not found, consume 1 part."

    Returns (parsed_value, num_parts_processed):
      - parsed_value: the found value or None if not found
      - num_parts_processed: how many parts were processed (1 or more)
    """

    if not attribute_parts_unprocessed:
        return (None, 0)

    key = attribute_parts_unprocessed[
        0
    ]  # If key is an int, then it represents a list index
    num_parts_processed = (
        1  # By default, we're at least consuming this first part
    )

    # 1) Try integer index (e.g. "0" => 0)
    idx = _try_int(key)
    if idx is not None:
        # Must be a tuple or list (_ensure_deserialized() already casts numpy arrays to python lists)
        if isinstance(current_value, (list, tuple)):
            if 0 <= idx < len(current_value):
                return (current_value[idx], num_parts_processed)
            else:
                return (None, num_parts_processed)
        else:
            return (None, num_parts_processed)

    # 2) Try dict approach
    if isinstance(current_value, dict):
        # a) direct match
        if key in current_value:
            return (current_value[key], num_parts_processed)
        else:
            # b) try combining multiple parts to handle dotted key
            for num_parts_processed in range(
                1, len(attribute_parts_unprocessed)
            ):
                key += "." + attribute_parts_unprocessed[num_parts_processed]
                if key in current_value:
                    return (
                        current_value[key],
                        num_parts_processed + 1,
                    )
            return (None, num_parts_processed)

    # If we get here, we couldn't handle it (not a list or dict or mismatch)
    return (None, num_parts_processed)


def _ensure_deserialized(val: Any) -> Any:
    """
    1) If `val` is a numpy array, convert to a Python list.
    2) If `val` is a string, attempt to parse as JSON.
    3) Otherwise return as-is.
    """
    if isinstance(val, np.ndarray):
        val = val.tolist()

    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return val


def _try_int(s: str) -> Optional[int]:
    """Attempt to convert s to int, return None on failure."""
    try:
        return int(s)
    except ValueError:
        return None
