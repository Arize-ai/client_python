"""DataFrame form validation for span annotations."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import pandas as pd

from arize.logging import log_a_list

# Import annotation-specific and common column constants
from arize.spans.columns import (
    ANNOTATION_COLUMN_PATTERN,
    ANNOTATION_COLUMN_PREFIX,
    ANNOTATION_LABEL_SUFFIX,
    ANNOTATION_NOTES_COLUMN_NAME,
    ANNOTATION_SCORE_SUFFIX,
    ANNOTATION_UPDATED_AT_SUFFIX,
    ANNOTATION_UPDATED_BY_SUFFIX,
    SPAN_SPAN_ID_COL,
)
from arize.spans.conversion import is_missing_value
from arize.spans.validation.common.errors import (
    InvalidAnnotationColumnFormat,
    InvalidDataFrameColumnContentTypes,
)

if TYPE_CHECKING:
    from arize.exceptions.base import ValidationError

logger = logging.getLogger(__name__)


def log_info_dataframe_extra_column_names(
    df: pd.DataFrame,
) -> None:
    """Logs columns that don't match expected annotation or context patterns."""
    if df is None:
        return
    # Check against annotation pattern, span id, and note column
    irrelevant_columns = [
        col
        for col in df.columns
        if not (
            pd.Series(col).str.match(ANNOTATION_COLUMN_PATTERN).any()
            or col == SPAN_SPAN_ID_COL.name
            or col == ANNOTATION_NOTES_COLUMN_NAME
        )
    ]
    if irrelevant_columns:
        logger.warning(
            "The following columns do not follow the annotation column naming convention "
            f"and will be ignored: {log_a_list(values=irrelevant_columns, join_word='and')}. "
            "Annotation columns must be named as follows: "
            "- annotation.<your-annotation-name>.label"
            "- annotation.<your-annotation-name>.score"
            f"An optional '{ANNOTATION_NOTES_COLUMN_NAME}' column can also be included."
        )
    return


def check_invalid_annotation_column_names(
    df: pd.DataFrame,
) -> list[ValidationError]:
    """Checks for columns that start with 'annotation.' but don't match the expected pattern."""
    errors: list[ValidationError] = []

    invalid_annotation_columns = [
        col
        for col in df.columns
        if col.startswith(ANNOTATION_COLUMN_PREFIX)
        and not pd.Series(col).str.match(ANNOTATION_COLUMN_PATTERN).any()
        and col != ANNOTATION_NOTES_COLUMN_NAME
    ]

    if invalid_annotation_columns:
        errors.append(
            InvalidAnnotationColumnFormat(
                invalid_format_cols=invalid_annotation_columns,
                expected_format="annotation.<name>.label|score|updated_by|updated_at",
            )
        )

    return errors


def check_dataframe_column_content_type(
    df: pd.DataFrame,
) -> list[ValidationError]:
    """Checks that columns matching annotation patterns have the correct data types."""
    wrong_labels_cols = []
    wrong_scores_cols = []
    wrong_notes_cols = []  # Add list for note column type errors
    wrong_updated_by_cols = []
    wrong_updated_at_cols = []
    errors = []

    # First check if there are any invalid annotation column names
    column_format_errors = check_invalid_annotation_column_names(df)
    if column_format_errors:
        errors.extend(column_format_errors)

    # Regex patterns for annotation suffixes
    annotation_label_re = re.compile(
        rf".+{re.escape(ANNOTATION_LABEL_SUFFIX)}$"
    )
    annotation_score_re = re.compile(
        rf".+{re.escape(ANNOTATION_SCORE_SUFFIX)}$"
    )
    annotation_updated_by_re = re.compile(
        rf".+{re.escape(ANNOTATION_UPDATED_BY_SUFFIX)}$"
    )
    annotation_updated_at_re = re.compile(
        rf".+{re.escape(ANNOTATION_UPDATED_AT_SUFFIX)}$"
    )

    for column in df.columns:
        # Check span ID column type (string)
        if column == SPAN_SPAN_ID_COL.name and not all(
            isinstance(value, str) for value in df[column]
        ):
            errors.append(
                InvalidDataFrameColumnContentTypes(
                    invalid_type_cols=[SPAN_SPAN_ID_COL.name],
                    expected_type="string",
                ),
            )
        # Check annotation label column type (string or missing)
        elif annotation_label_re.match(column):
            if not all(
                isinstance(value, str) or is_missing_value(value)
                for value in df[column]
            ):
                wrong_labels_cols.append(column)
        # Check annotation score column type (numeric or missing)
        elif annotation_score_re.match(column):
            if not all(
                isinstance(value, (int, float)) or is_missing_value(value)
                for value in df[column]
            ):
                wrong_scores_cols.append(column)
        # Check note column type (string or missing)
        elif column == ANNOTATION_NOTES_COLUMN_NAME:
            if not all(
                # Note: After formatting, this column holds list<string> (JSON), not just string.
                # We rely on later schema inference/validation. Keep basic check for now.
                isinstance(value, list) or is_missing_value(value)
                for value in df[column]
            ):
                wrong_notes_cols.append(column)
        # Check annotation updated_by column type (string or missing)
        elif annotation_updated_by_re.match(column):
            if not all(
                isinstance(value, str) or is_missing_value(value)
                for value in df[column]
            ):
                wrong_updated_by_cols.append(column)
        # Check annotation updated_at column type (numeric or missing)
        elif annotation_updated_at_re.match(column) and not all(
            # Allow int, float (e.g., Unix timestamp millis)
            isinstance(value, (int, float)) or is_missing_value(value)
            for value in df[column]
        ):
            wrong_updated_at_cols.append(column)

    # Append errors for wrong types
    if wrong_labels_cols:
        errors.append(
            InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_labels_cols,
                expected_type="strings",
            ),
        )
    if wrong_scores_cols:
        errors.append(
            InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_scores_cols,
                expected_type="ints or floats",
            ),
        )
    if wrong_notes_cols:
        errors.append(
            InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_notes_cols,
                expected_type="strings",
            ),
        )
    if wrong_updated_by_cols:
        errors.append(
            InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_updated_by_cols,
                expected_type="strings",
            ),
        )
    if wrong_updated_at_cols:
        errors.append(
            InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_updated_at_cols,
                expected_type="ints or floats (Unix timestamp)",
            ),
        )
    return errors
