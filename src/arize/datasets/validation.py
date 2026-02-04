"""Dataset validation logic for structure and content checks."""

import pandas as pd

from arize.datasets import errors as err


def validate_dataset_df(
    df: pd.DataFrame,
) -> list[err.DatasetError]:
    """Validate a dataset :class:`pandas.DataFrame` for structural and content errors.

    Checks for required columns, unique ID values, and non-empty data.

    Args:
        df: The :class:`pandas.DataFrame` to validate.

    Returns:
        A list of DatasetError objects found during validation. Empty list if valid.
    """
    ## check all require columns are present
    required_columns_errors = _check_required_columns(df)
    if required_columns_errors:
        return required_columns_errors

    ## check id column is unique
    id_column_unique_constraint_error = _check_id_column_is_unique(df)
    if id_column_unique_constraint_error:
        return id_column_unique_constraint_error

    # check DataFrame has at least one row in it
    empty_dataframe_error = _check_empty_dataframe(df)
    if empty_dataframe_error:
        return empty_dataframe_error

    return []


def _check_required_columns(df: pd.DataFrame) -> list[err.DatasetError]:
    required_columns = ["id", "created_at", "updated_at"]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        return [err.RequiredColumnsError(missing_columns)]
    return []


def _check_id_column_is_unique(df: pd.DataFrame) -> list[err.DatasetError]:
    if not df["id"].is_unique:
        return [err.IDColumnUniqueConstraintError()]
    return []


def _check_empty_dataframe(df: pd.DataFrame) -> list[err.DatasetError]:
    if df.empty:
        return [err.EmptyDatasetError()]
    return []
