"""DataFrame manipulation and validation utilities."""

import re

import pandas as pd

from arize.ml.types import BaseSchema


# Resets the dataframe index if it is not a RangeIndex
def reset_dataframe_index(dataframe: pd.DataFrame) -> None:
    """Reset the :class:`pandas.DataFrame` index in-place if it is not a RangeIndex.

    Args:
        dataframe: The :class:`pandas.DataFrame` to reset.
    """
    if not isinstance(dataframe.index, pd.RangeIndex):
        drop = dataframe.index.name in dataframe.columns
        dataframe.reset_index(inplace=True, drop=drop)


def remove_extraneous_columns(
    df: pd.DataFrame,
    schema: BaseSchema | None = None,
    column_list: list[str] | None = None,
    regex: str | None = None,
) -> pd.DataFrame:
    """Filter :class:`pandas.DataFrame` to keep only relevant columns based on schema, list, or regex.

    Args:
        df: The :class:`pandas.DataFrame` to filter.
        schema: Optional schema defining used columns. Defaults to None.
        column_list: Optional explicit list of columns to keep. Defaults to None.
        regex: Optional regex pattern to match column names. Defaults to None.

    Returns:
        A filtered DataFrame containing only the relevant columns.
    """
    relevant_columns = set()
    if schema is not None:
        relevant_columns.update(schema.get_used_columns())
    if column_list is not None:
        relevant_columns.update(column_list)
    if regex is not None:
        matched_regex_cols = []
        for col in df.columns:
            match_result = re.match(regex, col)
            if match_result:
                matched_regex_cols.append(col)
        relevant_columns.update(matched_regex_cols)

    final_columns = list(set(df.columns) & relevant_columns)
    return df.filter(items=final_columns)
