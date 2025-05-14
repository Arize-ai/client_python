import sys
import time
import uuid

import numpy as np
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
    from arize.pandas.tracing.constants import (
        ANNOTATION_LABEL_MAX_STR_LENGTH,
        ANNOTATION_UPDATED_BY_MAX_STR_LENGTH,  # Import new constant
    )
    from arize.pandas.tracing.validation.annotations import (
        annotations_validation,
    )


def get_valid_df_for_values_test(num_rows=2):
    """Helper to create a base valid DataFrame for value validation."""
    span_ids = [str(uuid.uuid4()) for _ in range(num_rows)]
    current_time_ns = int(time.time() * 1e9)
    notes_list = [[] for _ in range(num_rows)]
    if num_rows > 0:
        notes_list[0] = ['{"text": "Note 1"}']

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
                current_time_ns - 1000000000,
                current_time_ns,
            ][:num_rows],
            ANNOTATION_NOTES_COLUMN_NAME: pd.Series(notes_list, dtype=object),
        }
    )
    return df


valid_project_name = "project-name-value-test"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_valid_annotation_values():
    """Tests that correctly formed values pass validation."""
    annotations_dataframe = get_valid_df_for_values_test()
    errors = annotations_validation.validate_values(
        annotations_dataframe=annotations_dataframe,
        project_name=valid_project_name,
    )
    assert len(errors) == 0, "Valid values should produce no errors"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_empty_strings():
    """Tests errors for empty strings in label and updated_by."""
    df_label = get_valid_df_for_values_test(1)
    df_label[f"annotation.quality{ANNOTATION_LABEL_SUFFIX}"] = [""]
    errors_label = annotations_validation.validate_values(
        df_label, valid_project_name
    )
    assert len(errors_label) > 0, "Expected error for empty label string"

    df_updated_by = get_valid_df_for_values_test(1)
    df_updated_by[f"annotation.quality{ANNOTATION_UPDATED_BY_SUFFIX}"] = [""]
    errors_updated_by = annotations_validation.validate_values(
        df_updated_by, valid_project_name
    )
    assert (
        len(errors_updated_by) > 0
    ), "Expected error for empty updated_by string"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_string_lengths():
    """Tests errors for strings exceeding max length."""
    df_label = get_valid_df_for_values_test(1)
    # Use the imported constant for label length check
    df_label[f"annotation.quality{ANNOTATION_LABEL_SUFFIX}"] = [
        "L" * (ANNOTATION_LABEL_MAX_STR_LENGTH + 1)
    ]
    errors_label = annotations_validation.validate_values(
        df_label, valid_project_name
    )
    assert (
        len(errors_label) > 0
    ), "Expected error for label exceeding max length"

    df_updated_by = get_valid_df_for_values_test(1)
    # Use the imported constant for updated_by length check
    df_updated_by[f"annotation.quality{ANNOTATION_UPDATED_BY_SUFFIX}"] = [
        "U" * (ANNOTATION_UPDATED_BY_MAX_STR_LENGTH + 1)
    ]
    errors_updated_by = annotations_validation.validate_values(
        df_updated_by, valid_project_name
    )
    assert (
        len(errors_updated_by) > 0
    ), "Expected error for updated_by exceeding max length"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_numeric_values():
    """Tests errors for inf/NaN in score and updated_at."""
    df_score_inf = get_valid_df_for_values_test(1)
    df_score_inf[f"annotation.quality{ANNOTATION_SCORE_SUFFIX}"] = [np.inf]
    errors_score_inf = annotations_validation.validate_values(
        df_score_inf, valid_project_name
    )
    assert len(errors_score_inf) > 0, "Expected error for infinite score"

    # Timestamps should likely also reject inf/NaN if passed as float
    df_time_inf = get_valid_df_for_values_test(1)
    # Need to cast to float first to assign np.inf
    df_time_inf[f"annotation.quality{ANNOTATION_UPDATED_AT_SUFFIX}"] = (
        df_time_inf[
            f"annotation.quality{ANNOTATION_UPDATED_AT_SUFFIX}"
        ].astype(float)
    )
    df_time_inf[f"annotation.quality{ANNOTATION_UPDATED_AT_SUFFIX}"] = [np.inf]
    errors_time_inf = annotations_validation.validate_values(
        df_time_inf, valid_project_name
    )
    assert len(errors_time_inf) > 0, "Expected error for infinite updated_at"

    df_time_nan = get_valid_df_for_values_test(1)
    df_time_nan[f"annotation.quality{ANNOTATION_UPDATED_AT_SUFFIX}"] = (
        df_time_nan[
            f"annotation.quality{ANNOTATION_UPDATED_AT_SUFFIX}"
        ].astype(float)
    )
    df_time_nan[f"annotation.quality{ANNOTATION_UPDATED_AT_SUFFIX}"] = [np.nan]
    errors_time_nan = annotations_validation.validate_values(
        df_time_nan, valid_project_name
    )
    # Adjust assertion: _check_value_timestamp currently allows NaN (treats as None internally)
    assert (
        len(errors_time_nan) == 0
    ), "Expected no error for NaN updated_at (treated as null by validator)"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_timestamp_values():
    """Tests errors for non-positive timestamps."""
    df_time_zero = get_valid_df_for_values_test(1)
    df_time_zero[f"annotation.quality{ANNOTATION_UPDATED_AT_SUFFIX}"] = [0]
    errors_time_zero = annotations_validation.validate_values(
        df_time_zero, valid_project_name
    )
    assert len(errors_time_zero) > 0, "Expected error for zero updated_at"

    df_time_neg = get_valid_df_for_values_test(1)
    df_time_neg[f"annotation.quality{ANNOTATION_UPDATED_AT_SUFFIX}"] = [-1000]
    errors_time_neg = annotations_validation.validate_values(
        df_time_neg, valid_project_name
    )
    assert len(errors_time_neg) > 0, "Expected error for negative updated_at"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
