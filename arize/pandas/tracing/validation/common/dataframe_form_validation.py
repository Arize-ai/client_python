from typing import List

import pandas as pd
from arize.pandas.tracing.validation.common import errors as tracing_err
from arize.pandas.validation import errors as err


def _check_dataframe_index(dataframe: pd.DataFrame) -> List[err.InvalidDataFrameIndex]:
    if (dataframe.index != dataframe.reset_index(drop=True).index).any():
        return [err.InvalidDataFrameIndex()]
    return []


def _check_dataframe_required_column_set(
    df: pd.DataFrame,
    required_columns: List[str],
) -> List[tracing_err.InvalidDataFrameMissingColumns]:
    existing_columns = set(df.columns)
    missing_cols = []
    for col in required_columns:
        if col not in existing_columns:
            missing_cols.append(col)

    if missing_cols:
        return [tracing_err.InvalidDataFrameMissingColumns(missing_cols=missing_cols)]
    return []


def _check_dataframe_for_duplicate_columns(
    df: pd.DataFrame,
) -> List[tracing_err.InvalidDataFrameDuplicateColumns]:
    # Get the duplicated column names from the dataframe
    duplicate_columns = df.columns[df.columns.duplicated()]
    if not duplicate_columns.empty:
        return [tracing_err.InvalidDataFrameDuplicateColumns(duplicate_columns)]
    return []
