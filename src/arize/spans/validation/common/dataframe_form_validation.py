"""Common DataFrame form validation for spans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arize.exceptions.base import InvalidDataFrameIndex
from arize.spans.validation.common.errors import (
    InvalidDataFrameDuplicateColumns,
    InvalidDataFrameMissingColumns,
)

if TYPE_CHECKING:
    import pandas as pd


def check_dataframe_index(
    dataframe: pd.DataFrame,
) -> list[InvalidDataFrameIndex]:
    """Validates that the :class:`pandas.DataFrame` has a default integer index.

    Args:
        dataframe: The :class:`pandas.DataFrame` to validate.

    Returns:
        List of validation errors if index is not default (empty if valid).
    """
    if (dataframe.index != dataframe.reset_index(drop=True).index).any():
        return [InvalidDataFrameIndex()]
    return []


def check_dataframe_required_column_set(
    df: pd.DataFrame,
    required_columns: list[str],
) -> list[InvalidDataFrameMissingColumns]:
    """Validates that the :class:`pandas.DataFrame` contains all required columns.

    Args:
        df: The :class:`pandas.DataFrame` to validate.
        required_columns: List of column names that must be present.

    Returns:
        List of validation errors for missing columns (empty if valid).
    """
    existing_columns = set(df.columns)
    missing_cols = [
        col for col in required_columns if col not in existing_columns
    ]

    if missing_cols:
        return [InvalidDataFrameMissingColumns(missing_cols=missing_cols)]
    return []


def check_dataframe_for_duplicate_columns(
    df: pd.DataFrame,
) -> list[InvalidDataFrameDuplicateColumns]:
    """Validates that the :class:`pandas.DataFrame` has no duplicate column names.

    Args:
        df: The :class:`pandas.DataFrame` to validate.

    Returns:
        List of validation errors if duplicate columns exist (empty if valid).
    """
    # Get the duplicated column names from the dataframe
    duplicate_columns = df.columns[df.columns.duplicated()]
    if not duplicate_columns.empty:
        return [InvalidDataFrameDuplicateColumns(duplicate_columns.tolist())]
    return []
