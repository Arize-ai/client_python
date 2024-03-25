import re
from datetime import datetime
from itertools import chain
from typing import Iterable, List, Optional

import pandas as pd
from arize.pandas.tracing.columns import (
    EVAL_COLUMN_PATTERN,
    EVAL_EXPLANATION_PATTERN,
    EVAL_LABEL_PATTERN,
    EVAL_SCORE_PATTERN,
    SPAN_OPENINFERENCE_COLUMNS,
    SPAN_OPENINFERENCE_REQUIRED_COLUMNS,
    SPAN_SPAN_ID_COL,
    SpanColumnDataType,
)
from arize.pandas.tracing.utils import isMissingValue
from arize.pandas.tracing.validation import errors as tracing_err
from arize.pandas.validation import errors as err
from arize.utils.logging import log_a_list, logger
from arize.utils.types import is_array_of, is_dict_of, is_list_of
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def validate_dataframe_form(
    spans_dataframe: pd.DataFrame,
    evals_dataframe: Optional[pd.DataFrame] = None,
) -> List[err.ValidationError]:
    _info_dataframe_extra_column_names(spans_dataframe)
    _info_evals_dataframe_extra_column_names(evals_dataframe)
    checks = chain(
        _check_dataframe_index(spans_dataframe),
        _check_dataframe_required_column_set(
            spans_dataframe,
            required_columns=[col.name for col in SPAN_OPENINFERENCE_REQUIRED_COLUMNS],
        ),
        _check_dataframe_for_duplicate_columns(spans_dataframe),
        _check_dataframe_column_content_type(spans_dataframe),
    )
    if evals_dataframe is not None:
        checks = chain(
            checks,
            _check_dataframe_index(evals_dataframe),
            _check_dataframe_required_column_set(
                evals_dataframe, required_columns=[SPAN_SPAN_ID_COL.name]
            ),
            _check_dataframe_for_duplicate_columns(evals_dataframe),
            _check_evals_dataframe_column_content_type(evals_dataframe),
        )
    return list(checks)


def _check_dataframe_index(dataframe: pd.DataFrame) -> List[err.InvalidDataFrameIndex]:
    if (dataframe.index != dataframe.reset_index(drop=True).index).any():
        return [err.InvalidDataFrameIndex()]
    return []


def _info_dataframe_extra_column_names(
    df: pd.DataFrame,
) -> None:
    min_col_set = [col.name for col in SPAN_OPENINFERENCE_COLUMNS]
    extra_col_names = [col for col in df.columns if col not in min_col_set]
    if extra_col_names:
        logger.info(
            "The following columns are not part of the Open Inference Specification "
            f"and will be ignored: {log_a_list(list_of_str=extra_col_names, join_word='and')}"
        )
    return None


def _check_dataframe_required_column_set(
    df: pd.DataFrame,
    required_columns: List[str],
) -> List[tracing_err.InvalidDataFrameMissingColumns]:
    existing_columns = set(df.columns)
    missing_cols = []
    for col in required_columns:
        if col not in existing_columns:
            missing_cols.append(col)

    if missing_cols:
        return [tracing_err.InvalidDataFrameMissingColumns(missing_cols=missing_cols)]
    return []


def _check_dataframe_for_duplicate_columns(
    df: pd.DataFrame,
) -> List[tracing_err.InvalidDataFrameDuplicateColumns]:
    # Get the duplicated column names from the dataframe
    duplicate_columns = df.columns[df.columns.duplicated()]
    if not duplicate_columns.empty:
        return [tracing_err.InvalidDataFrameDuplicateColumns(duplicate_columns)]
    return []


# TODO(Kiko): Performance improvements
# We should try using:
# - Pandas any() and all() functions together with apply(), or
# - A combination of the following type checker functions from Pandas, i.e,
#   is_float_dtype. See link below
# https://github.com/pandas-dev/pandas/blob/f538741432edf55c6b9fb5d0d496d2dd1d7c2457/pandas/core/dtypes/common.py
def _check_dataframe_column_content_type(
    df: pd.DataFrame,
) -> List[tracing_err.InvalidDataFrameColumnContentTypes]:
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
                if not isinstance(row, Iterable) and isMissingValue(row):
                    continue
                if not (is_list_of(row, dict) or is_array_of(row, dict)) or not all(
                    is_dict_of(val, key_allowed_types=str) for val in row
                ):
                    wrong_lists_of_dicts_cols.append(col.name)
                    break
        elif col.data_type == SpanColumnDataType.DICT:
            if not all(
                (True if isMissingValue(row) else is_dict_of(row, key_allowed_types=str))
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
            # Accept strings and datetime objects
            if not all(
                (True if isMissingValue(row) else isinstance(row, (str, datetime, pd.Timestamp)))
                for row in df[col.name]
            ):
                wrong_timestamp_cols.append(col.name)
        elif col.data_type == SpanColumnDataType.JSON:
            # We check the correctness of the JSON strings when we check the values
            # of the data in the dataframe
            if not all(
                (True if isMissingValue(row) else isinstance(row, str)) for row in df[col.name]
            ):
                wrong_JSON_cols.append(col.name)
        elif col.data_type == SpanColumnDataType.STRING:
            if not all(
                (True if isMissingValue(row) else isinstance(row, str)) for row in df[col.name]
            ):
                wrong_string_cols.append(col.name)

    errors = []
    if wrong_lists_of_dicts_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_lists_of_dicts_cols,
                expected_type="lists of dictionaries with string keys",
            ),
        )
    if wrong_dicts_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_dicts_cols,
                expected_type="dictionaries with string keys",
            ),
        )
    if wrong_numeric_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_numeric_cols,
                expected_type="ints or floats",
            ),
        )
    if wrong_bools_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_bools_cols,
                expected_type="bools",
            ),
        )
    if wrong_timestamp_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_timestamp_cols,
                expected_type="datetime objects or formatted strings",
            ),
        )
    if wrong_JSON_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_JSON_cols,
                expected_type="JSON strings",
            ),
        )
    if wrong_string_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_string_cols,
                expected_type="strings",
            ),
        )
    return errors


def _info_evals_dataframe_extra_column_names(
    df: pd.DataFrame,
) -> None:
    if df is None:
        return None
    irrelevant_columns = [
        col
        for col in df.columns
        if not (pd.Series(col).str.match(EVAL_COLUMN_PATTERN).any() or col == SPAN_SPAN_ID_COL.name)
    ]
    if irrelevant_columns:
        logger.info(
            "The following columns do not follow the evaluation column naming convention "
            f"and will be ignored: {log_a_list(list_of_str=irrelevant_columns, join_word='and')}"
        )
    return None


def _check_evals_dataframe_column_content_type(
    df: pd.DataFrame,
) -> List[tracing_err.InvalidDataFrameColumnContentTypes]:
    wrong_labels_cols = []
    wrong_scores_cols = []
    wrong_explanations_cols = []
    errors = []
    eval_label_re = re.compile(EVAL_LABEL_PATTERN)
    eval_score_re = re.compile(EVAL_SCORE_PATTERN)
    eval_explanation_re = re.compile(EVAL_EXPLANATION_PATTERN)
    for column in df.columns:
        if column == SPAN_SPAN_ID_COL.name:
            if not all(isinstance(value, str) for value in df[column]):
                errors.append(
                    tracing_err.InvalidDataFrameColumnContentTypes(
                        invalid_type_cols=[SPAN_SPAN_ID_COL.name],
                        expected_type="string",
                    ),
                )
        if eval_label_re.match(column):
            if not all(isinstance(value, str) or isMissingValue(value) for value in df[column]):
                wrong_labels_cols.append(column)
        elif eval_score_re.match(column):
            if not all(
                isinstance(value, (int, float)) or isMissingValue(value) for value in df[column]
            ):
                wrong_scores_cols.append(column)
        elif eval_explanation_re.match(column):
            if not all(isinstance(value, str) or isMissingValue(value) for value in df[column]):
                wrong_explanations_cols.append(column)

    if wrong_labels_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_labels_cols,
                expected_type="strings",
            ),
        )
    if wrong_scores_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_scores_cols,
                expected_type="ints or floats",
            ),
        )
    if wrong_explanations_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_explanations_cols,
                expected_type="strings",
            ),
        )
    return errors
