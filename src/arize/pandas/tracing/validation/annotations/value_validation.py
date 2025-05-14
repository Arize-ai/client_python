import re
from itertools import chain
from typing import List

import pandas as pd

import arize.pandas.tracing.constants as tracing_constants

# Import annotation-specific and common column constants
from arize.pandas.tracing.columns import (
    ANNOTATION_COLUMN_PREFIX,
    ANNOTATION_LABEL_SUFFIX,
    ANNOTATION_NAME_PATTERN,
    ANNOTATION_NOTES_COLUMN_NAME,
    ANNOTATION_SCORE_SUFFIX,
    ANNOTATION_UPDATED_AT_SUFFIX,
    ANNOTATION_UPDATED_BY_SUFFIX,
)

# Import common validation errors and functions
from arize.pandas.tracing.validation.common import errors as tracing_err
from arize.pandas.tracing.validation.common import (
    value_validation as common_value_validation,
)
from arize.pandas.validation import errors as err


def _check_annotation_cols(
    dataframe: pd.DataFrame,
) -> List[err.ValidationError]:
    """Checks value length and validity for columns matching annotation patterns."""
    checks = []
    for col in dataframe.columns:
        if col.endswith(ANNOTATION_LABEL_SUFFIX):
            checks.append(
                common_value_validation._check_string_column_value_length(
                    df=dataframe,
                    col_name=col,
                    min_len=tracing_constants.ANNOTATION_LABEL_MIN_STR_LENGTH,
                    max_len=tracing_constants.ANNOTATION_LABEL_MAX_STR_LENGTH,
                    is_required=False,  # Individual columns are not required, null check handles completeness
                )
            )
        elif col.endswith(ANNOTATION_SCORE_SUFFIX):
            checks.append(
                common_value_validation._check_float_column_valid_numbers(
                    df=dataframe,
                    col_name=col,
                )
            )
        elif col.endswith(ANNOTATION_UPDATED_BY_SUFFIX):
            checks.append(
                common_value_validation._check_string_column_value_length(
                    df=dataframe,
                    col_name=col,
                    min_len=1,
                    max_len=tracing_constants.ANNOTATION_UPDATED_BY_MAX_STR_LENGTH,
                    is_required=False,
                )
            )
        elif col.endswith(ANNOTATION_UPDATED_AT_SUFFIX):
            checks.append(
                common_value_validation._check_value_timestamp(
                    df=dataframe,
                    col_name=col,
                    is_required=False,  # updated_at is not strictly required per row
                )
            )
        # No check for ANNOTATION_NOTES_COLUMN_NAME here, handled by _check_annotation_notes_column
    return list(chain(*checks))


def _check_annotation_columns_null_values(
    dataframe: pd.DataFrame,
) -> List[err.ValidationError]:
    """Checks that for a given annotation name, at least one of label or score is non-null per row."""
    invalid_annotation_names = []
    annotation_names = set()
    # Find all unique annotation names from column headers
    for col in dataframe.columns:
        match = re.match(ANNOTATION_NAME_PATTERN, col)
        if match:
            annotation_names.add(match.group(1))

    for ann_name in annotation_names:
        label_col = (
            f"{ANNOTATION_COLUMN_PREFIX}{ann_name}{ANNOTATION_LABEL_SUFFIX}"
        )
        score_col = (
            f"{ANNOTATION_COLUMN_PREFIX}{ann_name}{ANNOTATION_SCORE_SUFFIX}"
        )

        label_exists = label_col in dataframe.columns
        score_exists = score_col in dataframe.columns

        # Check only if both label and score columns exist for this name
        # If only one exists, its presence is sufficient
        if label_exists and score_exists:
            # Find rows where BOTH label and score are null
            condition = (
                dataframe[label_col].isnull() & dataframe[score_col].isnull()
            )
            if condition.any():
                invalid_annotation_names.append(ann_name)
        # Check if only label exists but it's always null
        elif label_exists and not score_exists:
            if dataframe[label_col].isnull().all():
                invalid_annotation_names.append(ann_name)
        # Check if only score exists but it's always null
        elif not label_exists and score_exists:
            if dataframe[score_col].isnull().all():
                invalid_annotation_names.append(ann_name)

    # Use set to report each name only once
    unique_invalid_names = sorted(list(set(invalid_annotation_names)))
    if unique_invalid_names:
        return [
            tracing_err.InvalidNullAnnotationLabelAndScore(
                annotation_names=unique_invalid_names
            )
        ]
    return []


def _check_annotation_notes_column(
    dataframe: pd.DataFrame,
) -> List[err.ValidationError]:
    """Checks the value length for the optional annotation.notes column (raw string)."""
    col_name = ANNOTATION_NOTES_COLUMN_NAME
    if col_name in dataframe.columns:
        # Validate the length of the raw string
        return list(
            chain(
                *common_value_validation._check_string_column_value_length(
                    df=dataframe,
                    col_name=col_name,
                    min_len=0,  # Allow empty notes
                    max_len=tracing_constants.ANNOTATION_NOTES_MAX_STR_LENGTH,
                    is_required=False,
                )
            )
        )
    return []
