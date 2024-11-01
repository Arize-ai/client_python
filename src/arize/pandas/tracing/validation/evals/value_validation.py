import re
from itertools import chain
from typing import List

import pandas as pd

import arize.pandas.tracing.constants as tracing_constants
from arize.pandas.tracing.columns import (
    EVAL_COLUMN_PREFIX,
    EVAL_EXPLANATION_SUFFIX,
    EVAL_LABEL_SUFFIX,
    EVAL_NAME_PATTERN,
    EVAL_SCORE_SUFFIX,
)
from arize.pandas.tracing.validation.common import errors as tracing_err
from arize.pandas.tracing.validation.common import value_validation
from arize.pandas.validation import errors as err


def _check_eval_cols(
    dataframe: pd.DataFrame,
) -> List[err.ValidationError]:
    checks = []
    for col in dataframe.columns:
        if col.endswith(EVAL_LABEL_SUFFIX):
            checks.append(
                value_validation._check_string_column_value_length(
                    df=dataframe,
                    col_name=col,
                    min_len=tracing_constants.EVAL_LABEL_MIN_STR_LENGTH,
                    max_len=tracing_constants.EVAL_LABEL_MAX_STR_LENGTH,
                    is_required=False,
                )
            )
        elif col.endswith(EVAL_SCORE_SUFFIX):
            checks.append(
                value_validation._check_float_column_valid_numbers(
                    df=dataframe,
                    col_name=col,
                )
            )
        elif col.endswith(EVAL_EXPLANATION_SUFFIX):
            checks.append(
                value_validation._check_string_column_value_length(
                    df=dataframe,
                    col_name=col,
                    min_len=0,
                    max_len=tracing_constants.EVAL_EXPLANATION_MAX_STR_LENGTH,
                    is_required=False,
                )
            )
    return list(chain(*checks))


# Evals are valid if they are entirely null (no label, score, or explanation) since this
# represents a span without an eval. Evals are also valid if at least one of label or score
# is not null
def _check_eval_columns_null_values(
    dataframe: pd.DataFrame,
) -> List[err.ValidationError]:
    invalid_eval_names = []
    eval_names = set()
    for col in dataframe.columns:
        match = re.match(EVAL_NAME_PATTERN, col)
        if match:
            eval_names.add(match.group(1))

    for eval_name in eval_names:
        label_col = f"{EVAL_COLUMN_PREFIX}{eval_name}{EVAL_LABEL_SUFFIX}"
        score_col = f"{EVAL_COLUMN_PREFIX}{eval_name}{EVAL_SCORE_SUFFIX}"
        explanation_col = (
            f"{EVAL_COLUMN_PREFIX}{eval_name}{EVAL_EXPLANATION_SUFFIX}"
        )
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
                invalid_eval_names.append(eval_name)

    if invalid_eval_names:
        return [
            tracing_err.InvalidNullEvalLabelAndScore(
                eval_names=invalid_eval_names
            )
        ]
    return []
