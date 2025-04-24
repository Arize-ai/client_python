import re
from typing import List

import pandas as pd

# Import annotation-specific and common column constants
from arize.pandas.tracing.columns import (
    ANNOTATION_COLUMN_PATTERN,
    ANNOTATION_LABEL_SUFFIX,
    ANNOTATION_NOTES_COLUMN_NAME,
    ANNOTATION_SCORE_SUFFIX,
    ANNOTATION_UPDATED_AT_SUFFIX,
    ANNOTATION_UPDATED_BY_SUFFIX,
    SPAN_SPAN_ID_COL,
)
from arize.pandas.tracing.utils import isMissingValue
from arize.pandas.tracing.validation.common import errors as tracing_err
from arize.utils.logging import log_a_list, logger


def _log_info_dataframe_extra_column_names(
    df: pd.DataFrame,
) -> None:
    """Logs columns that don't match expected annotation or context patterns."""
    if df is None:
        return None
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
        logger.info(
            "The following columns do not follow the annotation column naming convention "
            f"and will be ignored: {log_a_list(list_of_str=irrelevant_columns, join_word='and')}. "
            "Annotation columns must be named as follows: "
            "- annotation.<your-annotation-name>.label"
            "- annotation.<your-annotation-name>.score"
            f"An optional '{ANNOTATION_NOTES_COLUMN_NAME}' column can also be included."
        )
    return None


def _check_dataframe_column_content_type(
    df: pd.DataFrame,
) -> List[tracing_err.InvalidDataFrameColumnContentTypes]:
    """Checks that columns matching annotation patterns have the correct data types."""
    wrong_labels_cols = []
    wrong_scores_cols = []
    wrong_notes_cols = []  # Add list for note column type errors
    wrong_updated_by_cols = []
    wrong_updated_at_cols = []
    errors = []

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
                tracing_err.InvalidDataFrameColumnContentTypes(
                    invalid_type_cols=[SPAN_SPAN_ID_COL.name],
                    expected_type="string",
                ),
            )
        # Check annotation label column type (string or missing)
        elif annotation_label_re.match(column):
            if not all(
                isinstance(value, str) or isMissingValue(value)
                for value in df[column]
            ):
                wrong_labels_cols.append(column)
        # Check annotation score column type (numeric or missing)
        elif annotation_score_re.match(column):
            if not all(
                isinstance(value, (int, float)) or isMissingValue(value)
                for value in df[column]
            ):
                wrong_scores_cols.append(column)
        # Check note column type (string or missing)
        elif column == ANNOTATION_NOTES_COLUMN_NAME:
            if not all(
                # Note: After formatting, this column holds list<string> (JSON), not just string.
                # We rely on later schema inference/validation. Keep basic check for now.
                isinstance(value, list) or isMissingValue(value)
                for value in df[column]
            ):
                wrong_notes_cols.append(column)
        # Check annotation updated_by column type (string or missing)
        elif annotation_updated_by_re.match(column):
            if not all(
                isinstance(value, str) or isMissingValue(value)
                for value in df[column]
            ):
                wrong_updated_by_cols.append(column)
        # Check annotation updated_at column type (numeric or missing)
        elif annotation_updated_at_re.match(column) and not all(
            # Allow int, float (e.g., Unix timestamp millis)
            isinstance(value, (int, float)) or isMissingValue(value)
            for value in df[column]
        ):
            wrong_updated_at_cols.append(column)

    # Append errors for wrong types
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
    if wrong_notes_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_notes_cols,
                expected_type="strings",
            ),
        )
    if wrong_updated_by_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_updated_by_cols,
                expected_type="strings",
            ),
        )
    if wrong_updated_at_cols:
        errors.append(
            tracing_err.InvalidDataFrameColumnContentTypes(
                invalid_type_cols=wrong_updated_at_cols,
                expected_type="ints or floats (Unix timestamp)",
            ),
        )
    return errors
