"""Evaluation validation orchestration for spans."""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

from arize.spans.columns import SPAN_SPAN_ID_COL
from arize.spans.validation.common import (
    argument_validation as common_arg_validation,
)
from arize.spans.validation.common import (
    dataframe_form_validation as common_df_validation,
)
from arize.spans.validation.common import (
    value_validation as common_value_validation,
)
from arize.spans.validation.evals import (
    dataframe_form_validation as df_validation,
)
from arize.spans.validation.evals import (
    value_validation,
)

if TYPE_CHECKING:
    import pandas as pd

    from arize.exceptions.base import ValidationError


def validate_argument_types(
    evals_dataframe: pd.DataFrame,
    project_name: str,
    model_version: str | None = None,
) -> list[ValidationError]:
    """Validate argument types for evaluation data submission.

    Args:
        evals_dataframe: The DataFrame containing evaluation data.
        project_name: The project name to validate.
        model_version: Optional model version to validate. Defaults to None.

    Returns:
        List of validation errors found in argument types.
    """
    checks = chain(
        common_arg_validation.check_field_convertible_to_str(
            project_name, model_version
        ),
        common_arg_validation.check_dataframe_type(evals_dataframe),
    )
    return list(checks)


def validate_dataframe_form(
    evals_dataframe: pd.DataFrame,
) -> list[ValidationError]:
    """Validate the structure and form of an evaluations :class:`pandas.DataFrame`.

    Args:
        evals_dataframe: The :class:`pandas.DataFrame` containing evaluation data to validate.

    Returns:
        List of validation errors found in the :class:`pandas.DataFrame` structure.
    """
    df_validation.log_info_dataframe_extra_column_names(evals_dataframe)
    checks = chain(
        # Common
        common_df_validation.check_dataframe_index(evals_dataframe),
        common_df_validation.check_dataframe_required_column_set(
            evals_dataframe, required_columns=[SPAN_SPAN_ID_COL.name]
        ),
        common_df_validation.check_dataframe_for_duplicate_columns(
            evals_dataframe
        ),
        # Eval specific
        df_validation.check_dataframe_column_content_type(evals_dataframe),
    )
    return list(checks)


def validate_values(
    evals_dataframe: pd.DataFrame,
    project_name: str,
    model_version: str | None = None,
) -> list[ValidationError]:
    """Validate the values within an evaluations :class:`pandas.DataFrame`.

    Args:
        evals_dataframe: The :class:`pandas.DataFrame` containing evaluation data to validate.
        project_name: The project name associated with the evaluations.
        model_version: Optional model version. Defaults to None.

    Returns:
        List of validation errors found in :class:`pandas.DataFrame` values.
    """
    checks = chain(
        # Common
        common_value_validation.check_invalid_project_name(project_name),
        common_value_validation.check_invalid_model_version(model_version),
        # Eval specific
        value_validation.check_eval_cols(evals_dataframe),
        value_validation.check_eval_columns_null_values(evals_dataframe),
    )
    return list(checks)
