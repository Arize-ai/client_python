"""Common argument validation utilities for spans."""

import pandas as pd

from arize.exceptions.base import InvalidFieldTypeConversion
from arize.spans.validation.common.errors import InvalidTypeArgument


def check_field_convertible_to_str(
    project_name: object,
    model_version: object = None,
) -> list[InvalidFieldTypeConversion]:
    """Validates that field arguments can be converted to strings.

    Args:
        project_name: The project name value to validate for string conversion.
        model_version: Optional model version value to validate for string conversion.

    Returns:
        List of validation errors for fields that cannot be converted to strings.
    """
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
        return [InvalidFieldTypeConversion(wrong_fields, "string")]
    return []


def check_dataframe_type(
    dataframe: object,
) -> list[InvalidTypeArgument]:
    """Validates that the provided argument is a :class:`pandas.DataFrame`.

    Args:
        dataframe: The object to validate as a :class:`pandas.DataFrame`.

    Returns:
        List of validation errors if not a :class:`pandas.DataFrame` (empty if valid).
    """
    if not isinstance(dataframe, pd.DataFrame):
        return [
            InvalidTypeArgument(
                wrong_arg=dataframe,
                arg_name="dataframe",
                arg_type="pandas DataFrame",
            )
        ]
    return []


def check_datetime_format_type(
    dt_fmt: object,
) -> list[InvalidTypeArgument]:
    """Validates that the datetime format argument is a string.

    Args:
        dt_fmt: The datetime format value to validate.

    Returns:
        List of validation errors if not a string (empty if valid).
    """
    if not isinstance(dt_fmt, str):
        return [
            InvalidTypeArgument(
                wrong_arg=dt_fmt,
                arg_name="dateTime format",
                arg_type="string",
            )
        ]
    return []
