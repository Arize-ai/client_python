"""Annotation validation orchestration for spans."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

from arize.spans.columns import SPAN_SPAN_ID_COL
from arize.spans.validation.annotations import (
    dataframe_form_validation as df_validation,
)
from arize.spans.validation.annotations import value_validation
from arize.spans.validation.common import (
    argument_validation as common_arg_validation,
)
from arize.spans.validation.common import (
    dataframe_form_validation as common_df_validation,
)
from arize.spans.validation.common import (
    value_validation as common_value_validation,
)

if TYPE_CHECKING:
    import pandas as pd

    from arize.exceptions.base import ValidationError


def validate_argument_types(
    annotations_dataframe: pd.DataFrame,
    project_name: str,
) -> list[ValidationError]:
    """Validates argument types for log_annotations."""
    checks = chain(
        common_arg_validation.check_field_convertible_to_str(project_name),
        common_arg_validation.check_dataframe_type(
            annotations_dataframe
        ),  # Use renamed parameter
    )
    return list(checks)


def validate_dataframe_form(
    annotations_dataframe: pd.DataFrame,
) -> list[ValidationError]:
    """Validates the form/structure of the annotation dataframe."""
    # Call annotation-specific function (to be created)
    df_validation.log_info_dataframe_extra_column_names(annotations_dataframe)
    checks = chain(
        # Common checks remain the same
        common_df_validation.check_dataframe_index(annotations_dataframe),
        common_df_validation.check_dataframe_required_column_set(
            annotations_dataframe, required_columns=[SPAN_SPAN_ID_COL.name]
        ),
        common_df_validation.check_dataframe_for_duplicate_columns(
            annotations_dataframe
        ),
        # Call annotation-specific content type check (to be created)
        df_validation.check_dataframe_column_content_type(
            annotations_dataframe
        ),
    )
    return list(checks)


def validate_values(
    annotations_dataframe: pd.DataFrame,
    project_name: str,
) -> list[ValidationError]:
    """Validates the values within the annotation dataframe."""
    checks = chain(
        # Common checks remain the same
        common_value_validation.check_invalid_project_name(project_name),
        # Call annotation-specific value checks from the imported module
        value_validation.check_annotation_cols(annotations_dataframe),
        value_validation.check_annotation_columns_null_values(
            annotations_dataframe
        ),
        value_validation.check_annotation_notes_column(annotations_dataframe),
    )
    return list(checks)
