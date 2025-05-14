import sys
import time
import uuid

import pandas as pd
import pytest

if sys.version_info >= (3, 8):
    from arize.pandas.tracing.columns import (
        ANNOTATION_LABEL_SUFFIX,
        ANNOTATION_NOTES_COLUMN_NAME,
        ANNOTATION_SCORE_SUFFIX,
        ANNOTATION_UPDATED_AT_SUFFIX,
        ANNOTATION_UPDATED_BY_SUFFIX,
        SPAN_SPAN_ID_COL,
    )
    from arize.pandas.tracing.validation.annotations import (
        annotations_validation,
    )


def get_valid_annotation_df(num_rows=2):
    """Helper to create a DataFrame with the correct type for the notes column."""
    span_ids = [str(uuid.uuid4()) for _ in range(num_rows)]
    current_time_ms = int(time.time() * 1000)

    # Initialize notes with empty lists for all rows to ensure consistent list type
    notes_list = [[] for _ in range(num_rows)]
    if num_rows > 0:
        notes_list[0] = [
            '{"text": "Note 1"}'
        ]  # Assign the actual note to the first row

    df = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: span_ids,
            f"annotation.quality{ANNOTATION_LABEL_SUFFIX}": ["good", "bad"][
                :num_rows
            ],
            f"annotation.quality{ANNOTATION_SCORE_SUFFIX}": [0.9, 0.1][
                :num_rows
            ],
            f"annotation.quality{ANNOTATION_UPDATED_BY_SUFFIX}": [
                "user1",
                "user2",
            ][:num_rows],
            f"annotation.quality{ANNOTATION_UPDATED_AT_SUFFIX}": [
                current_time_ms - 1000,
                current_time_ms,
            ][:num_rows],
            # Use the pre-initialized list with consistent type
            ANNOTATION_NOTES_COLUMN_NAME: pd.Series(notes_list, dtype=object),
        }
    )
    return df


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_valid_annotation_column_types():
    """Tests that a DataFrame with correct types passes validation."""
    annotations_dataframe = get_valid_annotation_df()
    errors = annotations_validation.validate_dataframe_form(
        annotations_dataframe=annotations_dataframe
    )
    assert len(errors) == 0, "Expected no validation errors for valid types"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_annotation_label_type():
    """Tests error for non-string label column."""
    annotations_dataframe = get_valid_annotation_df()
    annotations_dataframe[f"annotation.quality{ANNOTATION_LABEL_SUFFIX}"] = [
        1,
        2,
    ]
    errors = annotations_validation.validate_dataframe_form(
        annotations_dataframe=annotations_dataframe
    )
    assert len(errors) > 0


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_annotation_score_type():
    """Tests error for non-numeric score column."""
    annotations_dataframe = get_valid_annotation_df()
    annotations_dataframe[f"annotation.quality{ANNOTATION_SCORE_SUFFIX}"] = [
        "high",
        "low",
    ]
    errors = annotations_validation.validate_dataframe_form(
        annotations_dataframe=annotations_dataframe
    )
    assert len(errors) > 0


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_annotation_updated_by_type():
    """Tests error for non-string updated_by column."""
    annotations_dataframe = get_valid_annotation_df()
    annotations_dataframe[
        f"annotation.quality{ANNOTATION_UPDATED_BY_SUFFIX}"
    ] = [100, 200]
    errors = annotations_validation.validate_dataframe_form(
        annotations_dataframe=annotations_dataframe
    )
    assert len(errors) > 0


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_annotation_updated_at_type():
    """Tests error for non-numeric updated_at column."""
    annotations_dataframe = get_valid_annotation_df()
    annotations_dataframe[
        f"annotation.quality{ANNOTATION_UPDATED_AT_SUFFIX}"
    ] = ["yesterday", "today"]
    errors = annotations_validation.validate_dataframe_form(
        annotations_dataframe=annotations_dataframe
    )
    assert len(errors) > 0


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_annotation_notes_type():
    """Tests error for non-list notes column."""
    annotations_dataframe = get_valid_annotation_df()
    annotations_dataframe[ANNOTATION_NOTES_COLUMN_NAME] = "just a string"
    errors = annotations_validation.validate_dataframe_form(
        annotations_dataframe=annotations_dataframe
    )
    assert len(errors) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
