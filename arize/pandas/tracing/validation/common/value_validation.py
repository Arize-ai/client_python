from datetime import datetime, timedelta
from typing import List, Optional, Union

import arize.pandas.tracing.columns as tracing_cols
import numpy as np
import pandas as pd
import pyarrow as pa
from arize.pandas.tracing.validation.common import errors as tracing_err
from arize.pandas.validation import errors as err
from arize.utils.constants import (
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
)
from arize.utils.logging import logger
from arize.utils.types import is_json_str


def _check_invalid_project_name(project_name: Optional[str]) -> List[err.InvalidProjectName]:
    # assume it's been coerced to string beforehand
    if (not isinstance(project_name, str)) or len(project_name.strip()) == 0:
        return [err.InvalidProjectName()]
    return []


def _check_invalid_model_version(
    model_version: Optional[str] = None,
) -> List[err.InvalidModelVersion]:
    if model_version is None:
        return []
    if not isinstance(model_version, str) or len(model_version.strip()) == 0:
        return [err.InvalidModelVersion()]

    return []


def _check_string_column_value_length(
    df: pd.DataFrame,
    col_name: str,
    min_len: int,
    max_len: int,
    is_required: bool,
    must_be_json: bool = False,
) -> List[Union[tracing_err.InvalidMissingValueInColumn, tracing_err.InvalidStringLengthInColumn]]:
    if col_name not in df.columns:
        return []

    errors = []
    if is_required and df[col_name].isnull().any():
        errors.append(
            tracing_err.InvalidMissingValueInColumn(
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
            tracing_err.InvalidStringLengthInColumn(
                col_name=col_name,
                min_length=min_len,
                max_length=max_len,
            )
        )
    if must_be_json and not df[~df[col_name].isnull()][col_name].apply(is_json_str).all():
        errors.append(tracing_err.InvalidJsonStringInColumn(col_name=col_name))

    return errors


def _check_string_column_allowed_values(
    df: pd.DataFrame,
    col_name: str,
    allowed_values: List[str],
    is_required: bool,
) -> List[
    Union[
        tracing_err.InvalidMissingValueInColumn,
        tracing_err.InvalidStringValueNotAllowedInColumn,
    ]
]:
    if col_name not in df.columns:
        return []

    errors = []
    if is_required and df[col_name].isnull().any():
        errors.append(
            tracing_err.InvalidMissingValueInColumn(
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
            tracing_err.InvalidStringValueNotAllowedInColumn(
                col_name=col_name,
                allowed_values=allowed_values,
            )
        )
    return errors


# Checks to make sure there are no inf values in the column
def _check_float_column_valid_numbers(
    df: pd.DataFrame,
    col_name: str,
) -> List[Union[tracing_err.InvalidFloatValueInColumn]]:
    if col_name not in df.columns:
        return []
    # np.isinf will fail on None values, change Nones to np.nan and check on that
    column_numeric = pd.to_numeric(df[col_name], errors="coerce")
    invalid_mask = np.isinf(column_numeric)
    invalid_exists = invalid_mask.any()

    if invalid_exists:
        error = [tracing_err.InvalidFloatValueInColumn(col_name=col_name)]
        return error
    return []


def _check_value_columns_start_end_time(
    df: pd.DataFrame,
) -> List[
    Union[
        tracing_err.InvalidMissingValueInColumn,
        tracing_err.InvalidTimestampValueInColumn,
        tracing_err.InvalidStartAndEndTimeValuesInColumn,
    ]
]:
    errors = []
    errors += _check_value_timestamp(
        df=df,
        col_name=tracing_cols.SPAN_START_TIME_COL.name,
        is_required=tracing_cols.SPAN_START_TIME_COL.required,
    )
    errors += _check_value_timestamp(
        df=df,
        col_name=tracing_cols.SPAN_END_TIME_COL.name,
        is_required=tracing_cols.SPAN_END_TIME_COL.required,
    )
    if (
        tracing_cols.SPAN_START_TIME_COL.name in df.columns
        and tracing_cols.SPAN_END_TIME_COL.name in df.columns
        and (
            df[tracing_cols.SPAN_START_TIME_COL.name] > df[tracing_cols.SPAN_END_TIME_COL.name]
        ).any()
    ):
        errors.append(
            tracing_err.InvalidStartAndEndTimeValuesInColumn(
                greater_col_name=tracing_cols.SPAN_END_TIME_COL.name,
                less_col_name=tracing_cols.SPAN_START_TIME_COL.name,
            )
        )
    return errors


def _check_value_timestamp(
    df: pd.DataFrame,
    col_name: str,
    is_required: bool,
) -> List[
    Union[
        tracing_err.InvalidMissingValueInColumn,
        tracing_err.InvalidTimestampValueInColumn,
    ]
]:
    # This check expects that timestamps have previously been converted to nanoseconds
    if col_name not in df.columns:
        return []

    errors = []
    if is_required and df[col_name].isnull().any():
        errors.append(
            tracing_err.InvalidMissingValueInColumn(
                col_name=col_name,
            )
        )

    now_t = datetime.now()
    lbound, ubound = (
        (now_t - timedelta(days=MAX_PAST_YEARS_FROM_CURRENT_TIME * 365)).timestamp() * 1e9,
        (now_t + timedelta(days=MAX_FUTURE_YEARS_FROM_CURRENT_TIME * 365)).timestamp() * 1e9,
    )

    # faster than pyarrow compute
    stats = df[col_name].agg(["min", "max"])

    ta = pa.Table.from_pandas(stats.to_frame())
    min_, max_ = ta.column(0)
    if max_.as_py() > now_t.timestamp() * 1e9:
        logger.warning(
            f"Detected future timestamp in column '{col_name}'. "
            "Caution when sending spans with future timestamps. "
            "Arize only stores 2 years worth of data. For example, if you sent spans "
            "to Arize from 1.5 years ago, and now send spans with timestamps of a year in "
            "the future, the oldest 0.5 years will be dropped to maintain the 2 years worth of data "
            "requirement."
        )

    if min_.as_py() < lbound or max_.as_py() > ubound:
        return [tracing_err.InvalidTimestampValueInColumn(timestamp_col_name=col_name)]

    return []
