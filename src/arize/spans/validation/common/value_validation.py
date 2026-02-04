"""Common value validation logic for span data."""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa

from arize.constants.ml import (
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
)
from arize.exceptions.base import ValidationError
from arize.exceptions.parameters import InvalidModelVersion, InvalidProjectName
from arize.spans.columns import (
    SPAN_END_TIME_COL,
    SPAN_START_TIME_COL,
)
from arize.spans.validation.common.errors import (
    InvalidFloatValueInColumn,
    InvalidJsonStringInColumn,
    InvalidMissingValueInColumn,
    InvalidStartAndEndTimeValuesInColumn,
    InvalidStringLengthInColumn,
    InvalidStringValueNotAllowedInColumn,
    InvalidTimestampValueInColumn,
)
from arize.utils.types import is_json_str

logger = logging.getLogger(__name__)


def check_invalid_project_name(
    project_name: str | None,
) -> list[InvalidProjectName]:
    """Validates that the project name is a non-empty string.

    Args:
        project_name: The project name to validate.

    Returns:
        List of validation errors if project name is invalid (empty if valid).
    """
    # assume it's been coerced to string beforehand
    if (not isinstance(project_name, str)) or len(project_name.strip()) == 0:
        return [InvalidProjectName()]
    return []


def check_invalid_model_version(
    model_version: str | None = None,
) -> list[InvalidModelVersion]:
    """Validates that the model version, if provided, is a non-empty string.

    Args:
        model_version: The optional model version to validate.

    Returns:
        List of validation errors if model version is invalid (empty if valid or :obj:`None`).
    """
    if model_version is None:
        return []
    if not isinstance(model_version, str) or len(model_version.strip()) == 0:
        return [InvalidModelVersion()]

    return []


def check_string_column_value_length(
    df: pd.DataFrame,
    col_name: str,
    min_len: int,
    max_len: int,
    is_required: bool,
    must_be_json: bool = False,
) -> list[ValidationError]:
    """Validate string column values are within length bounds and optionally valid JSON.

    Args:
        df: The DataFrame to validate.
        col_name: Name of the column to check.
        min_len: Minimum allowed string length.
        max_len: Maximum allowed string length.
        is_required: Whether the column must have non-null values.
        must_be_json: Whether values must be valid JSON strings. Defaults to False.

    Returns:
        List of validation errors for missing values, invalid lengths, or invalid JSON.
    """
    if col_name not in df.columns:
        return []

    errors: list[ValidationError] = []
    if is_required and df[col_name].isnull().any():
        errors.append(
            InvalidMissingValueInColumn(
                col_name=col_name,
            )
        )

    if not (
        # Check that the non-None values of the desired colum have a
        # string length between min_len and max_len
        # Does not check the None values
        df[~df[col_name].isnull()][col_name]
        .astype(str)
        .str.len()
        .between(min_len, max_len)
        .all()
    ):
        errors.append(
            InvalidStringLengthInColumn(
                col_name=col_name,
                min_length=min_len,
                max_length=max_len,
            )
        )
    if (
        must_be_json
        and not df[~df[col_name].isnull()][col_name].apply(is_json_str).all()
    ):
        errors.append(InvalidJsonStringInColumn(col_name=col_name))

    return errors


def check_string_column_allowed_values(
    df: pd.DataFrame,
    col_name: str,
    allowed_values: list[str],
    is_required: bool,
) -> list[ValidationError]:
    """Validate that string column values are within allowed values.

    Args:
        df: The DataFrame to validate.
        col_name: The column name to check.
        allowed_values: List of allowed string values (case-insensitive).
        is_required: Whether the column must not have missing values.

    Returns:
        List of validation errors found.
    """
    if col_name not in df.columns:
        return []

    errors: list[ValidationError] = []
    if is_required and df[col_name].isnull().any():
        errors.append(
            InvalidMissingValueInColumn(
                col_name=col_name,
            )
        )

    # We compare in lowercase
    allowed_values_lowercase = [v.lower() for v in allowed_values]
    if not (
        # Check that the non-None values of the desired colum have a
        # string values amongst the ones allowed
        # Does not check the None values
        df[~df[col_name].isnull()][col_name]
        .astype(str)
        .str.lower()
        .isin(allowed_values_lowercase)
        .all()
    ):
        errors.append(
            InvalidStringValueNotAllowedInColumn(
                col_name=col_name,
                allowed_values=allowed_values,
            )
        )
    return errors


# Checks to make sure there are no inf values in the column
def check_float_column_valid_numbers(
    df: pd.DataFrame,
    col_name: str,
) -> list[ValidationError]:
    """Check that float column contains only finite numbers, no infinity values.

    Args:
        df: The DataFrame to validate.
        col_name: The column name to check.

    Returns:
        List containing InvalidFloatValueInColumn error if infinite values found.
    """
    if col_name not in df.columns:
        return []
    # np.isinf will fail on None values, change Nones to np.nan and check on that
    column_numeric = pd.to_numeric(df[col_name], errors="coerce")
    invalid_mask = np.isinf(column_numeric)
    invalid_exists = invalid_mask.any()

    if invalid_exists:
        return [InvalidFloatValueInColumn(col_name=col_name)]
    return []


def check_value_columns_start_end_time(
    df: pd.DataFrame,
) -> list[ValidationError]:
    """Validate start and end time columns for timestamps and logical ordering.

    Args:
        df: The DataFrame containing start and end time columns.

    Returns:
        List of validation errors for missing values, invalid timestamps, or start > end.
    """
    errors: list[ValidationError] = []
    errors += check_value_timestamp(
        df=df,
        col_name=SPAN_START_TIME_COL.name,
        is_required=SPAN_START_TIME_COL.required,
    )
    errors += check_value_timestamp(
        df=df,
        col_name=SPAN_END_TIME_COL.name,
        is_required=SPAN_END_TIME_COL.required,
    )
    if (
        SPAN_START_TIME_COL.name in df.columns
        and SPAN_END_TIME_COL.name in df.columns
        and (df[SPAN_START_TIME_COL.name] > df[SPAN_END_TIME_COL.name]).any()
    ):
        errors.append(
            InvalidStartAndEndTimeValuesInColumn(
                greater_col_name=SPAN_END_TIME_COL.name,
                less_col_name=SPAN_START_TIME_COL.name,
            )
        )
    return errors


def check_value_timestamp(
    df: pd.DataFrame,
    col_name: str,
    is_required: bool,
) -> list[ValidationError]:
    """Validate timestamp column values are within reasonable bounds.

    Args:
        df: The DataFrame to validate.
        col_name: The column name containing timestamps in nanoseconds.
        is_required: Whether missing values should be flagged as errors.

    Returns:
        List of validation errors for missing or out-of-bounds timestamps.
    """
    # This check expects that timestamps have previously been converted to nanoseconds
    if col_name not in df.columns:
        return []

    errors: list[ValidationError] = []
    if is_required and df[col_name].isnull().any():
        errors.append(
            InvalidMissingValueInColumn(
                col_name=col_name,
            )
        )

    now_t = datetime.now(tz=timezone.utc)
    lbound, ubound = (
        (
            now_t - timedelta(days=MAX_PAST_YEARS_FROM_CURRENT_TIME * 365)
        ).timestamp()
        * 1e9,
        (
            now_t + timedelta(days=MAX_FUTURE_YEARS_FROM_CURRENT_TIME * 365)
        ).timestamp()
        * 1e9,
    )

    # faster than pyarrow compute
    stats = df[col_name].agg(["min", "max"])

    ta = pa.Table.from_pandas(stats.to_frame())
    min_, max_ = ta.column(0)

    # Check if min/max are None before comparing (handles NaN input)
    min_val = min_.as_py()
    max_val = max_.as_py()

    if max_val is not None and max_val > now_t.timestamp() * 1e9:
        logger.warning(
            f"Detected future timestamp in column '{col_name}'. "
            "Caution when sending spans with future timestamps. "
            "Arize only stores 2 years worth of data. For example, if you sent spans "
            "to Arize from 1.5 years ago, and now send spans with timestamps of a year in "
            "the future, the oldest 0.5 years will be dropped to maintain the 2 years worth of data "
            "requirement."
        )

    if (min_val is not None and min_val < lbound) or (
        max_val is not None and max_val > ubound
    ):
        return [InvalidTimestampValueInColumn(timestamp_col_name=col_name)]

    return []
