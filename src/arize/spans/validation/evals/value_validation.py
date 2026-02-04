"""Value validation logic for span evaluation data."""

from __future__ import annotations

import re
from itertools import chain
from typing import TYPE_CHECKING

from arize.constants.spans import (
    EVAL_EXPLANATION_MAX_STR_LENGTH,
    EVAL_LABEL_MAX_STR_LENGTH,
    EVAL_LABEL_MIN_STR_LENGTH,
)
from arize.spans.columns import (
    EVAL_EXPLANATION_SUFFIX,
    EVAL_LABEL_SUFFIX,
    EVAL_PREFIX_AND_NAME_PATTERN,
    EVAL_SCORE_SUFFIX,
)
from arize.spans.validation.common import value_validation
from arize.spans.validation.common.errors import InvalidNullEvalLabelAndScore

if TYPE_CHECKING:
    import pandas as pd

    from arize.exceptions.base import ValidationError


def check_eval_cols(
    dataframe: pd.DataFrame,
) -> list[ValidationError]:
    """Validates evaluation column values for proper length and format.

    Checks label strings for length constraints, scores for valid numeric values,
    and explanations for length constraints.

    Args:
        dataframe: The DataFrame containing evaluation columns.

    Returns:
        List of validation errors found in evaluation columns.
    """
    checks: list[list[ValidationError]] = []
    for col in dataframe.columns:
        if col.endswith(EVAL_LABEL_SUFFIX):
            checks.append(
                value_validation.check_string_column_value_length(
                    df=dataframe,
                    col_name=col,
                    min_len=EVAL_LABEL_MIN_STR_LENGTH,
                    max_len=EVAL_LABEL_MAX_STR_LENGTH,
                    is_required=False,
                )
            )
        elif col.endswith(EVAL_SCORE_SUFFIX):
            checks.append(
                value_validation.check_float_column_valid_numbers(
                    df=dataframe,
                    col_name=col,
                )
            )
        elif col.endswith(EVAL_EXPLANATION_SUFFIX):
            checks.append(
                value_validation.check_string_column_value_length(
                    df=dataframe,
                    col_name=col,
                    min_len=0,
                    max_len=EVAL_EXPLANATION_MAX_STR_LENGTH,
                    is_required=False,
                )
            )
    return list(chain(*checks))


# Evals are valid if they are entirely null (no label, score, or explanation) since this
# represents a span without an eval. Evals are also valid if at least one of label or score
# is not null
def check_eval_columns_null_values(
    dataframe: pd.DataFrame,
) -> list[ValidationError]:
    """Validates that evaluation columns don't have orphan explanations without labels or scores.

    Ensures that if an explanation exists, at least one of label or score is non-null.

    Args:
        dataframe: The DataFrame containing evaluation columns.

    Returns:
        List of validation errors for evaluations with invalid null combinations.
    """
    invalid_eval_names = []
    eval_prefix_and_name = set()
    for col in dataframe.columns:
        match = re.match(EVAL_PREFIX_AND_NAME_PATTERN, col)
        if match:
            # match is eval.Hallucination or session_eval.Repetitive
            eval_prefix_and_name.add(match.group(1))

    for prefix_and_name in eval_prefix_and_name:
        label_col = prefix_and_name + EVAL_LABEL_SUFFIX
        score_col = prefix_and_name + EVAL_SCORE_SUFFIX
        explanation_col = prefix_and_name + EVAL_EXPLANATION_SUFFIX
        columns_to_check = []

        if label_col in dataframe.columns:
            columns_to_check.append(label_col)
        if score_col in dataframe.columns:
            columns_to_check.append(score_col)

        # If there are explanations, they cannot be orphan ()
        if explanation_col in dataframe.columns:
            condition = (
                dataframe[columns_to_check].isnull().all(axis=1)
                & ~dataframe[explanation_col].isnull()
            )
            if condition.any():
                invalid_eval_names.append(prefix_and_name)

    if invalid_eval_names:
        return [InvalidNullEvalLabelAndScore(eval_names=invalid_eval_names)]
    return []
