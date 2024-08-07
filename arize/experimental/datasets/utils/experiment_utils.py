import dataclasses
import datetime
import functools
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

try:
    from typing import get_args, get_origin  # Python 3.8+
except ImportError:
    from typing_extensions import get_args, get_origin  # For Python <3.8


import numpy as np


def get_func_name(fn: Callable[..., Any]) -> str:
    """
    Makes a best-effort attempt to get the name of the function.
    """

    if isinstance(fn, functools.partial):
        return fn.func.__qualname__
    if hasattr(fn, "__qualname__") and not fn.__qualname__.endswith("<lambda>"):
        return fn.__qualname__.split(".<locals>.")[-1]
    return str(fn)


def jsonify(obj: Any) -> Any:
    """
    Coerce object to be json serializable.
    """
    if isinstance(obj, Enum):
        return jsonify(obj.value)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, set, frozenset, Sequence)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, (dict, Mapping)):
        return {jsonify(k): jsonify(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj):
        result = {}
        for field in dataclasses.fields(obj):
            k = field.name
            v = getattr(obj, k)
            if not (v is None and get_origin(field) is Union and type(None) in get_args(field)):
                result[k] = jsonify(v)
        return result
    if isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return obj.total_seconds()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BaseException):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return [jsonify(v) for v in obj]
    if hasattr(obj, "__float__"):
        return float(obj)
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        # pydantic v2
        try:
            d = obj
            assert isinstance(d, dict)
        except BaseException:
            pass
        else:
            return jsonify(d)
    if hasattr(obj, "dict") and callable(obj.dict):
        # pydantic v1
        try:
            d = obj.dict()
            assert isinstance(d, dict)
        except BaseException:
            pass
        else:
            return jsonify(d)
    cls = obj.__class__
    return f"<{cls.__module__}.{cls.__name__} object>"
