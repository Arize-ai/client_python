"""OpenInference data conversion utilities for column transformations."""

import json
import logging

import pandas as pd

from arize.constants.openinference import OPEN_INFERENCE_JSON_STR_TYPES

logger = logging.getLogger(__name__)


def convert_datetime_columns_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns in a :class:`pandas.DataFrame` to milliseconds since epoch.

    Args:
        df: The :class:`pandas.DataFrame` to convert.

    Returns:
        The :class:`pandas.DataFrame` with datetime columns converted to integers.
    """
    for col in df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ):
        df[col] = df[col].astype("int64") // 10**6  # ms since epoch
    return df


def convert_boolean_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean columns in a :class:`pandas.DataFrame` to string type.

    Args:
        df: The :class:`pandas.DataFrame` to convert.

    Returns:
        The :class:`pandas.DataFrame` with boolean columns converted to strings.
    """
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype("string")
    return df


def convert_default_columns_to_json_str(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dictionary values in specific columns to JSON strings.

    Args:
        df: The :class:`pandas.DataFrame` to convert.

    Returns:
        The :class:`pandas.DataFrame` with dictionaries in eligible columns converted to JSON strings.
    """
    for col in df.columns:
        if _should_convert_json(col):
            try:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )
            except Exception as e:
                logger.debug(
                    f"Failed to convert column '{col}' to JSON string: {e}"
                )
                continue
    return df


def convert_json_str_to_dict(df: pd.DataFrame) -> pd.DataFrame:
    """Convert JSON string values in specific columns to Python dictionaries.

    Args:
        df: The :class:`pandas.DataFrame` to convert.

    Returns:
        The :class:`pandas.DataFrame` with JSON strings in eligible columns converted to dictionaries.
    """
    for col in df.columns:
        if _should_convert_json(col):
            try:
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
            except Exception as e:
                logger.debug(f"Failed to parse column '{col}' as JSON: {e}")
                continue
    return df


def _should_convert_json(col_name: str) -> bool:
    """Check if a column should be converted to/from a JSON string/PythonDictionary."""
    is_eval_metadata = col_name.startswith("eval.") and col_name.endswith(
        ".metadata"
    )
    is_json_str = col_name in OPEN_INFERENCE_JSON_STR_TYPES
    is_task_result = col_name == "result"
    return is_eval_metadata or is_json_str or is_task_result
