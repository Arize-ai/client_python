from typing import Any, List

import pandas as pd

from arize.pandas.tracing.validation.common import errors as tracing_err
from arize.pandas.validation import errors as err


def _check_field_convertible_to_str(
    project_name: str,
    model_version: str,
) -> List[err.InvalidFieldTypeConversion]:
    wrong_fields = []
    if project_name is not None and not isinstance(project_name, str):
        try:
            str(project_name)
        except Exception:
            wrong_fields.append("project_name")
    if model_version is not None and not isinstance(model_version, str):
        try:
            str(model_version)
        except Exception:
            wrong_fields.append("model_version")

    if wrong_fields:
        return [err.InvalidFieldTypeConversion(wrong_fields, "string")]
    return []


def _check_dataframe_type(
    dataframe,
) -> List[tracing_err.InvalidTypeArgument]:
    if not isinstance(dataframe, pd.DataFrame):
        return [
            tracing_err.InvalidTypeArgument(
                wrong_arg=dataframe,
                arg_name="dataframe",
                arg_type="pandas DataFrame",
            )
        ]
    return []


def _check_datetime_format_type(
    dt_fmt: Any,
) -> List[tracing_err.InvalidTypeArgument]:
    if not isinstance(dt_fmt, str):
        return [
            tracing_err.InvalidTypeArgument(
                wrong_arg=dt_fmt,
                arg_name="dateTime format",
                arg_type="string",
            )
        ]
    return []
