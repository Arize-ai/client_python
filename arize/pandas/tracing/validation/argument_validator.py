from itertools import chain
from typing import Any, List, Optional

import pandas as pd
from arize.pandas.tracing.validation import errors as tracing_err
from arize.pandas.validation import errors as err


def validate_argument_types(
    dataframe: pd.DataFrame,
    model_id: str,
    dt_fmt: str,
    model_version: Optional[str] = None,
) -> List[err.ValidationError]:
    return list(
        chain(
            _check_field_convertible_to_str(model_id, model_version),
            _check_dataframe_type(dataframe),
            _check_datetime_format_type(dt_fmt),
        )
    )


def _check_field_convertible_to_str(
    model_id: str,
    model_version: str,
) -> List[err.InvalidFieldTypeConversion]:
    wrong_fields = []
    if model_id is not None and not isinstance(model_id, str):
        try:
            str(model_id)
        except Exception:
            wrong_fields.append("model_id")
    if model_version is not None and not isinstance(model_version, str):
        try:
            str(model_version)
        except Exception:
            wrong_fields.append("model_version")

    if wrong_fields:
        return [err.InvalidFieldTypeConversion(wrong_fields, "string")]
    return []


def _check_dataframe_type(
    dataframe: Any,
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
