"""Common type definitions and data models used across the Arize SDK."""

import json
from collections.abc import Iterable, Sequence
from typing import (
    TypeVar,
)

import numpy as np


def is_json_str(s: str) -> bool:
    """Check if a string is valid JSON.

    Args:
        s: The string to validate.

    Returns:
        True if the string is valid JSON, False otherwise.
    """
    try:
        json.loads(s)
    except ValueError:
        return False
    except TypeError:
        return False
    return True


T = TypeVar("T", bound=type)


def is_array_of(arr: Sequence[object], tp: T) -> bool:
    """Check if a value is a numpy array with all elements of a specific type.

    Args:
        arr: The sequence to check.
        tp: The expected type for all elements.

    Returns:
        True if arr is a numpy array and all elements are of type tp.
    """
    return isinstance(arr, np.ndarray) and all(isinstance(x, tp) for x in arr)


def is_list_of(lst: object, tp: T) -> bool:
    """Check if a value is a list with all elements of a specific type.

    Args:
        lst: The sequence to check.
        tp: The expected type for all elements.

    Returns:
        True if lst is a list and all elements are of type tp.
    """
    return isinstance(lst, list) and all(isinstance(x, tp) for x in lst)


def is_iterable_of(lst: Sequence[object], tp: T) -> bool:
    """Check if a value is an iterable with all elements of a specific type.

    Args:
        lst: The sequence to check.
        tp: The expected type for all elements.

    Returns:
        True if lst is an iterable and all elements are of type tp.
    """
    return isinstance(lst, Iterable) and all(isinstance(x, tp) for x in lst)


def is_dict_of(
    d: object,
    key_allowed_types: type | tuple[type, ...],
    value_allowed_types: type | tuple[type, ...] = (),
    value_list_allowed_types: type | tuple[type, ...] = (),
) -> bool:
    """Method to check types are valid for dictionary.

    Args:
        d: Dictionary itself.
        key_allowed_types: All allowed types for keys of dictionary.
        value_allowed_types: All allowed types for values of dictionary.
        value_list_allowed_types: If value is a list, these are the allowed
            types for value list.

    Returns:
        True if the data types of dictionary match the types specified by the
            arguments, false otherwise.
    """
    if value_list_allowed_types and not isinstance(
        value_list_allowed_types, tuple
    ):
        value_list_allowed_types = (value_list_allowed_types,)

    return (
        isinstance(d, dict)
        and all(isinstance(k, key_allowed_types) for k in d)
        and all(
            isinstance(v, value_allowed_types)
            or any(is_list_of(v, t) for t in value_list_allowed_types)  # type: ignore[union-attr]
            for v in d.values()
            if value_allowed_types or value_list_allowed_types
        )
    )
