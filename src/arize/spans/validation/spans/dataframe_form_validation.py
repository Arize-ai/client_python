"""DataFrame form validation for spans."""

import logging
from collections.abc import Iterable
from datetime import datetime

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from arize.spans.columns import SPAN_OPENINFERENCE_COLUMNS, SpanColumnDataType
from arize.spans.conversion import is_missing_value
from arize.spans.validation.common.errors import (
    InvalidDataFrameColumnContentTypes,
)
from arize.utils.types import is_array_of, is_dict_of, is_list_of

logger = logging.getLogger(__name__)


def log_info_dataframe_extra_column_names(
    df: pd.DataFrame,
) -> None:
    """Logs informational message about columns not part of Open Inference Specification.

    Args:
        df: DataFrame to check for extra column names.

    Returns:
        None.
    """
    min_col_set = [col.name for col in SPAN_OPENINFERENCE_COLUMNS]
    extra_col_names = [col for col in df.columns if col not in min_col_set]
    if extra_col_names:
        logger.info(
            "Found columns that are not part of the Open Inference Specification "
            "and will be ignored",
            extra={
                "extra_columns": extra_col_names,
            },
        )
    return


# TODO(Kiko): Performance improvements
# We should try using:
# - Pandas any() and all() functions together with apply(), or
# - A combination of the following type checker functions from Pandas, i.e,
#   is_float_dtype. See link below
# https://github.com/pandas-dev/pandas/blob/f538741432edf55c6b9fb5d0d496d2dd1d7c2457/pandas/core/dtypes/common.py
def check_dataframe_column_content_type(
    df: pd.DataFrame,
) -> list[InvalidDataFrameColumnContentTypes]:
    """Validates span :class:`pandas.DataFrame` columns match OpenInference types.

    Checks that columns have appropriate data types: lists of dicts, dicts, numeric,
    boolean, timestamp, JSON strings, or plain strings based on column specifications.

    Args:
        df: The :class:`pandas.DataFrame` to validate.

    Returns:
        List of validation errors for columns with incorrect types.
    """
    # We let this values be in the dataframe and don't use them to verify type
    # They will be serialized by arrow and understood as missing values
    wrong_lists_of_dicts_cols = []
    wrong_dicts_cols = []
    wrong_numeric_cols = []
    wrong_bools_cols = []
    wrong_timestamp_cols = []
    wrong_JSON_cols = []
    wrong_string_cols = []
    for col in SPAN_OPENINFERENCE_COLUMNS:
        if col.name not in df.columns:
            continue
        if col.data_type == SpanColumnDataType.LIST_DICT:
            for row in df[col.name]:
                if not isinstance(row, Iterable) and is_missing_value(row):
                    continue
                if not (
                    is_list_of(row, dict) or is_array_of(row, dict)
                ) or not all(
                    is_dict_of(val, key_allowed_types=str) for val in row
                ):
                    wrong_lists_of_dicts_cols.append(col.name)
                    break
        elif col.data_type == SpanColumnDataType.DICT:
            if not all(
                (
                    is_missing_value(row)
                    or is_dict_of(row, key_allowed_types=str)
                )
                for row in df[col.name]
            ):
                wrong_dicts_cols.append(col.name)
        elif col.data_type == SpanColumnDataType.NUMERIC:
            if not is_numeric_dtype(df[col.name]):
                wrong_numeric_cols.append(col.name)
        elif col.data_type == SpanColumnDataType.BOOL:
            if not is_bool_dtype(df[col.name]):
                wrong_bools_cols.append(col.name)
        elif col.data_type == SpanColumnDataType.TIMESTAMP:
            # Accept strings and datetime objects, and int64
            if not all(
                (
                    is_missing_value(row)
                    or isinstance(row, (str, datetime, pd.Timestamp, int))
                )
                for row in df[col.name]
            ):
                wrong_timestamp_cols.append(col.name)
        elif col.data_type == SpanColumnDataType.JSON:
            # We check the correctness of the JSON strings when we check the values
            # of the data in the dataframe
            if not all(
                (is_missing_value(row) or isinstance(row, str))
                for row in df[col.name]
            ):
                wrong_JSON_cols.append(col.name)
        elif col.data_type == SpanColumnDataType.STRING and not all(
            (is_missing_value(row) or isinstance(row, str))
            for row in df[col.name]
        ):
            wrong_string_cols.append(col.name)

    errors = []
    if wrong_lists_of_dicts_cols:
        errors.append(
            InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_lists_of_dicts_cols,
                expected_type="lists of dictionaries with string keys",
            ),
        )
    if wrong_dicts_cols:
        errors.append(
            InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_dicts_cols,
                expected_type="dictionaries with string keys",
            ),
        )
    if wrong_numeric_cols:
        errors.append(
            InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_numeric_cols,
                expected_type="ints or floats",
            ),
        )
    if wrong_bools_cols:
        errors.append(
            InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_bools_cols,
                expected_type="bools",
            ),
        )
    if wrong_timestamp_cols:
        errors.append(
            InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_timestamp_cols,
                expected_type="datetime objects or formatted strings or integers (nanoseconds)",
            ),
        )
    if wrong_JSON_cols:
        errors.append(
            InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_JSON_cols,
                expected_type="JSON strings",
            ),
        )
    if wrong_string_cols:
        errors.append(
            InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_string_cols,
                expected_type="strings",
            ),
        )
    return errors
