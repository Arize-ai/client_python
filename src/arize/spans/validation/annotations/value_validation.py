"""Value validation logic for span annotation data."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from itertools import chain
from typing import TYPE_CHECKING

from arize.constants.spans import (
    ANNOTATION_LABEL_MAX_STR_LENGTH,
    ANNOTATION_LABEL_MIN_STR_LENGTH,
    ANNOTATION_NOTES_MAX_STR_LENGTH,
    ANNOTATION_UPDATED_BY_MAX_STR_LENGTH,
)
from arize.exceptions.base import ValidationError

# Import annotation-specific and common column constants
from arize.spans.columns import (
    ANNOTATION_COLUMN_PREFIX,
    ANNOTATION_LABEL_SUFFIX,
    ANNOTATION_NAME_PATTERN,
    ANNOTATION_NOTES_COLUMN_NAME,
    ANNOTATION_SCORE_SUFFIX,
    ANNOTATION_UPDATED_AT_SUFFIX,
    ANNOTATION_UPDATED_BY_SUFFIX,
)
from arize.spans.validation.common import (
    value_validation as common_value_validation,
)

# Import common validation errors and functions
from arize.spans.validation.common.errors import (
    InvalidMissingValueInColumn,
    InvalidNullAnnotationLabelAndScore,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class InvalidAnnotationTimestamp(ValidationError):
    """Raised when annotation timestamp is invalid or out of acceptable range."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Annotation_Timestamp"

    def __init__(self, timestamp_col_name: str, error_type: str) -> None:
        """Initialize the exception with timestamp validation context.

        Args:
            timestamp_col_name: Name of the annotation timestamp column.
            error_type: Type of timestamp error (e.g., 'future').
        """
        self.timestamp_col_name = timestamp_col_name
        self.error_type = error_type

    def error_message(self) -> str:
        """Return the error message for this exception."""
        if self.error_type == "future":
            return (
                f"At least one timestamp in the annotation column '{self.timestamp_col_name}' "
                f"is in the future. Annotation timestamps cannot be in the future."
            )
        if self.error_type == "non_positive":
            return (
                f"At least one timestamp in the annotation column '{self.timestamp_col_name}' "
                f"is zero or negative. Annotation timestamps must be positive values."
            )
        return f"Invalid timestamp in annotation column '{self.timestamp_col_name}'."


def check_annotation_updated_at_timestamp(
    df: pd.DataFrame,
    col_name: str,
    is_required: bool,
) -> list[ValidationError]:
    """Validates annotation timestamp values for validity and acceptable ranges.

    Checks that timestamp values are positive, not in the future, and satisfy
    required constraints if specified.

    Args:
        df: DataFrame containing the annotation timestamp column.
        col_name: Name of the timestamp column to validate.
        is_required: Whether the column must have non-null values in all rows.

    Returns:
        List of validation errors found (empty if valid).
    """
    # This check expects that timestamps have previously been converted to milliseconds
    if col_name not in df.columns:
        return []

    errors: list[ValidationError] = []
    if is_required and df[col_name].isnull().any():
        errors.append(
            InvalidMissingValueInColumn(
                col_name=col_name,
            )
        )

    if df[col_name].isnull().all():
        return errors

    now_ms = datetime.now(tz=timezone.utc).timestamp() * 1000

    if df[col_name].max() > now_ms:
        logger.warning(f"Detected future timestamp in column '{col_name}'.")
        errors.append(
            InvalidAnnotationTimestamp(
                timestamp_col_name=col_name, error_type="future"
            )
        )

    if df[col_name].min() <= 0:
        errors.append(
            InvalidAnnotationTimestamp(
                timestamp_col_name=col_name, error_type="non_positive"
            )
        )

    return errors


def check_annotation_cols(
    dataframe: pd.DataFrame,
) -> list[ValidationError]:
    """Checks value length and validity for columns matching annotation patterns."""
    checks: list[list[ValidationError]] = []
    for col in dataframe.columns:
        if col.endswith(ANNOTATION_LABEL_SUFFIX):
            checks.append(
                common_value_validation.check_string_column_value_length(
                    df=dataframe,
                    col_name=col,
                    min_len=ANNOTATION_LABEL_MIN_STR_LENGTH,
                    max_len=ANNOTATION_LABEL_MAX_STR_LENGTH,
                    # Individual columns are not required
                    is_required=False,
                )
            )
        elif col.endswith(ANNOTATION_SCORE_SUFFIX):
            checks.append(
                common_value_validation.check_float_column_valid_numbers(
                    df=dataframe,
                    col_name=col,
                )
            )
        elif col.endswith(ANNOTATION_UPDATED_BY_SUFFIX):
            checks.append(
                common_value_validation.check_string_column_value_length(
                    df=dataframe,
                    col_name=col,
                    min_len=1,
                    max_len=ANNOTATION_UPDATED_BY_MAX_STR_LENGTH,
                    is_required=False,
                )
            )
        elif col.endswith(ANNOTATION_UPDATED_AT_SUFFIX):
            checks.append(
                check_annotation_updated_at_timestamp(
                    df=dataframe,
                    col_name=col,
                    is_required=False,  # updated_at is not strictly required per row
                )
            )
        # No check for ANNOTATION_NOTES_COLUMN_NAME here, handled by check_annotation_notes_column
    return list(chain(*checks))


def check_annotation_columns_null_values(
    dataframe: pd.DataFrame,
) -> list[ValidationError]:
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
    unique_invalid_names = sorted(set(invalid_annotation_names))
    if unique_invalid_names:
        return [
            InvalidNullAnnotationLabelAndScore(
                annotation_names=unique_invalid_names
            )
        ]
    return []


def check_annotation_notes_column(
    dataframe: pd.DataFrame,
) -> list[ValidationError]:
    """Checks the value length for the optional annotation.notes column (raw string)."""
    col_name = ANNOTATION_NOTES_COLUMN_NAME
    if col_name in dataframe.columns:
        # Validate the length of the raw string
        return common_value_validation.check_string_column_value_length(
            df=dataframe,
            col_name=col_name,
            min_len=0,  # Allow empty notes
            max_len=ANNOTATION_NOTES_MAX_STR_LENGTH,
            is_required=False,
        )
    return []
