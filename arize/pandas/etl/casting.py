from typing import List, Tuple

import numpy as np
import pandas as pd
from arize.utils.logging import log_a_list
from arize.utils.types import Schema, TypedColumns, is_list_of

from .errors import ColumnCastingError, InvalidSchemaFieldTypeError, InvalidTypedColumnsError

ETL_MINIMUM_PANDAS_VERSION = "1.0.0"
ETL_ERROR_MESSAGE = (
    f"To enable ETL, upgrade pandas to version {ETL_MINIMUM_PANDAS_VERSION} or higher."
)


def validate_typed_columns(field_name: str, typed_columns: TypedColumns) -> None:
    """
    Validate a TypedColumns object.

    Arguments
    ----------
        field_name: str
            The name of the Schema field that the TypedColumns object is associated with.
        typed_columns: TypedColumns
            The TypedColumns object to validate.
    Raises
    ----------
        InvalidTypedColumnsError
            If the TypedColumns object is invalid.
    """
    if typed_columns.is_empty():
        raise InvalidTypedColumnsError(field_name=field_name, reason="is empty")
    has_duplicates, duplicates = typed_columns.has_duplicate_columns()
    if has_duplicates:
        raise InvalidTypedColumnsError(
            field_name=field_name,
            reason=f"has duplicate column names: {log_a_list(list(duplicates), 'and')}",
        )


def cast_typed_columns(dataframe: pd.DataFrame, schema: Schema) -> Tuple[pd.DataFrame, Schema]:
    """
    Cast feature and tag columns in the dataframe to the types specified in each TypedColumns config.
    This optional feature provides a simple way for users to prevent
    type drift within a column across many SDK uploads.

    Arguments:
    ----------
        dataframe: pd.DataFrame
            A deepcopy of the user's dataframe.
        schema: Schema
            The schema, which may include feature and tag column names
            in a TypedColumns object or a List[string].

    Returns:
    ----------
        dataframe: pd.DataFrame
            The dataframe, with columns cast to the specified types.
        schema: Schema
            A new Schema object, with feature and tag column names converted to the List[string] format
            expected in downstream validation.

    Raises:
    ----------
        ColumnCastingError
            If casting fails.
        InvalidTypedColumnsError
            If the TypedColumns object is invalid.
    """
    typed_column_fields = schema.typed_column_fields()
    feature_field = "feature_column_names"
    tag_field = "tag_column_names"
    allowed_fields = {feature_field, tag_field}

    # Make sure the schema has typed column fields.
    if not typed_column_fields:
        raise InvalidSchemaFieldTypeError(
            "The Schema object does not have any fields of type TypedColumns. "
            "Cannot cast dataframe columns."
        )

    # Make sure no other schema fields have this type.
    if any({f for f in typed_column_fields if f not in allowed_fields}):
        raise InvalidSchemaFieldTypeError(
            "Only the feature_column_names and tag_column_names Schema fields can be of type "
            "TypedColumns. Fields with type TypedColumns:" + str(typed_column_fields)
        )

    for field_name in typed_column_fields:
        f = getattr(schema, field_name)
        if f:
            try:
                validate_typed_columns(field_name, f)
            except InvalidTypedColumnsError:
                raise
            dataframe = cast_columns(dataframe, f)

    # Now that the dataframe values have been cast to the specified types:
    # for downstream validation to work as expected,
    # feature & tag schema field types should be List[string] of column names.
    # Since Schema is a frozen class, we must construct a new instance.
    return dataframe, convert_schema_field_types(schema)


def cast_columns(dataframe: pd.DataFrame, columns: TypedColumns) -> pd.DataFrame:
    """
    Cast columns corresponding to a single TypedColumns object and a single Arize Schema field.
    (feature_column_names or tag_column_names)

    Arguments:
    ----------
        dataframe: pd.DataFrame
            A deepcopy of the user's dataframe.
        columns: TypedColumns
            The TypedColumns object, which specifies the columns to cast
            (and/or to not cast) and their target types.

    Returns:
    ----------
        dataframe: pd.DataFrame
            The dataframe with columns cast to the specified types.

    Raises:
    ----------
        ColumnCastingError
            If casting fails.
    """
    if columns.to_str:
        try:
            # Nullable StringDtype is an experimental feature:
            # https://pandas.pydata.org/docs/reference/api/pandas.StringDtype.html
            # https://pandas.pydata.org/docs/user_guide/text.html#working-with-text-data
            # 'string' is an alias for StringDtype
            # uses pd.NA for missing values (when storage arg is not configured)
            # In the future, try out pd.convert_dtypes (new in pandas 2.0):
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.convert_dtypes.html
            dataframe = cast_df(dataframe, columns.to_str, "string")
        except Exception as e:
            raise ColumnCastingError(
                error_msg=str(e),
                attempted_columns=columns.to_str,
                attempted_type="string",
            )
    if columns.to_int:
        # pandas nullable type must be capitalized: 'Int64'
        # see https://pandas.pydata.org/docs/reference/api/pandas.Int64Dtype.html
        # uses pd.NA for missing values
        try:
            dataframe = cast_df(dataframe, columns.to_int, "Int64")
        except Exception as e:
            raise ColumnCastingError(
                error_msg=str(e),
                attempted_columns=columns.to_int,
                attempted_type="Int64",
            )
    if columns.to_float:
        # pandas nullable type must be capitalized: 'Float64'
        # see https://pandas.pydata.org/docs/reference/api/pandas.Float64Dtype.html
        # uses pd.NA for missing values
        try:
            dataframe = cast_df(dataframe, columns.to_float, "Float64")
        except Exception as e:
            raise ColumnCastingError(
                error_msg=str(e),
                attempted_columns=columns.to_float,
                attempted_type="Float64",
            )

    return dataframe


def cast_df(df: pd.DataFrame, cols: List[str], type_str: str) -> pd.DataFrame:
    """
    Arguments:
    ----------
        df: pd.DataFrame
            A deepcopy of the user's dataframe.
        cols: List[str]
            The list of column names to cast.
        type_str: str
            The target type to cast to.

    Returns:
    ----------
        df: pd.DataFrame
            The dataframe with columns cast to the specified types.

    Raises:
    ----------
        Exception
            If casting fails. Common exceptions raised by astype() are TypeError and ValueError.
    """
    if type_str == "string":
        # when NaN floats are cast to string, they become "nan". Replace with actual NaN values.
        return df.astype({col: type_str for col in cols}).replace("nan", np.nan)

    # todo (Hannah): in the future do we want to have "NaN"s in string columns convert to np.nan?
    return df.astype({col: type_str for col in cols})


def convert_schema_field_types(
    schema: Schema,
) -> Schema:
    """
    Arguments:
    ----------
        schema: Schema
            The schema, which may include feature and tag column names
            in a TypedColumns object or a List[string].

    Returns:
    ----------
        schema: Schema
            A Schema, with feature and tag column names
            converted to the List[string] format expected in downstream validation.
    """
    feature_column_names_list = (
        schema.feature_column_names
        if is_list_of(schema.feature_column_names, str)
        else (
            schema.feature_column_names.get_all_column_names()
            if schema.feature_column_names
            else []
        )
    )

    tag_column_names_list = (
        schema.tag_column_names
        if is_list_of(schema.tag_column_names, str)
        else schema.tag_column_names.get_all_column_names()
        if schema.tag_column_names
        else []
    )

    schema_dict = {
        "feature_column_names": feature_column_names_list,
        "tag_column_names": tag_column_names_list,
    }
    return schema.replace(**schema_dict)
