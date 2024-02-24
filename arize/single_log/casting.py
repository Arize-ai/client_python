import math
from typing import List, Union

from arize.utils.types import ArizeTypes, TypedValue

from .errors import CastingError


def cast_dictionary(d: dict) -> Union[dict, None]:
    if not d:
        return None
    cast_dict = {}
    for k, v in d.items():
        if isinstance(v, TypedValue):
            v = cast_value(v)
        cast_dict[k] = v
    return cast_dict


def cast_value(typed_value: TypedValue) -> Union[str, int, float, List[str], None]:
    """
    Casts a TypedValue to its provided type, preserving all null values as None or float('nan').

    Arguments:
    ----------
    typed_value: TypedValue
        The TypedValue to cast.

    Returns:
    --------
    Union[str, int, float, List[str], None]
        The cast value.

    Raises:
    -------
    CastingError
        If the value cannot be cast to the provided type.
    """
    if typed_value.value is None:
        return None

    if typed_value.type == ArizeTypes.FLOAT:
        return _cast_to_float(typed_value)
    elif typed_value.type == ArizeTypes.INT:
        return _cast_to_int(typed_value)
    elif typed_value.type == ArizeTypes.STR:
        return _cast_to_str(typed_value)
    else:
        raise CastingError("Unknown casting type", typed_value)


def _cast_to_float(typed_value: TypedValue) -> Union[float, None]:
    try:
        return float(typed_value.value)
    except Exception as e:
        raise CastingError(str(e), typed_value)


def _cast_to_int(typed_value: TypedValue) -> Union[int, None]:
    # a NaN float can't be cast to an int. Proactively return None instead.
    if isinstance(typed_value.value, float) and math.isnan(typed_value.value):
        return None
    # If the value is a float, to avoid losing data precision,
    # we can only cast to an int if it is equivalent to an integer (e.g. 7.0).
    if isinstance(typed_value.value, float) and not typed_value.value.is_integer():
        raise CastingError("Cannot convert float with non-zero fractional part to int", typed_value)
    try:
        return int(typed_value.value)
    except Exception as e:
        raise CastingError(str(e), typed_value)


def _cast_to_str(typed_value: TypedValue) -> Union[str, None]:
    # a NaN float can't be cast to a string. Proactively return None instead.
    if isinstance(typed_value.value, float) and math.isnan(typed_value.value):
        return None
    try:
        return str(typed_value.value)
    except Exception as e:
        raise CastingError(str(e), typed_value)
