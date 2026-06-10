"""Dataset validation logic for structure and content checks."""

import pandas as pd

from arize.datasets import errors as err


def validate_dataset_df(
    df: pd.DataFrame,
) -> list[err.DatasetError]:
    """Validate a dataset :class:`pandas.DataFrame` for structural and content errors.

    Checks for binary columns, required columns, unique ID values, and
    non-empty data.

    Args:
        df: The :class:`pandas.DataFrame` to validate.

    Returns:
        A list of DatasetError objects found during validation. Empty list if valid.
    """
    # reject bytes-valued columns before any network call; they are encoded as
    # a binary column type the dataset service cannot store as text.
    binary_column_errors = _check_binary_columns(df)
    if binary_column_errors:
        return binary_column_errors

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


def _check_binary_columns(df: pd.DataFrame) -> list[err.DatasetError]:
    """Return a :class:`BinaryColumnError` listing every column containing bytes.

    Python ``bytes`` values are encoded as a binary column type that the
    dataset service cannot store as text. Callers should pass text as ``str``
    instead. All offending columns are collected so the user gets the full
    list at once rather than fixing them one re-run at a time.

    ``dtype`` alone can't classify a column: pandas stores both ``str`` and
    ``bytes`` as ``object``. We use it only as a cheap prefilter (numeric,
    bool, and datetime columns can't hold bytes) and then inspect the values.
    All non-null values are checked, not just the first: pyarrow infers an
    Arrow ``binary`` column if *any* value is bytes, so a column whose first
    value is a ``str`` but later values are ``bytes`` would still be rejected
    server-side.
    """
    binary_columns = [
        col
        for col in df.columns
        if df[col].dtype == object
        and any(isinstance(v, (bytes, bytearray)) for v in df[col].dropna())
    ]
    if binary_columns:
        return [err.BinaryColumnError(binary_columns)]
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
