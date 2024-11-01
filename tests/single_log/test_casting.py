import math
import sys
from dataclasses import dataclass
from typing import List, Union

import pytest

from arize.single_log.casting import cast_value
from arize.single_log.errors import CastingError
from arize.utils.types import ArizeTypes, TypedValue


@dataclass
class SingleLogTestCase:
    typed_value: TypedValue
    expected_value: Union[int, float, str, None]
    expected_error: CastingError = None


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_string_to_float_error():
    tv = TypedValue(value="fruit", type=ArizeTypes.FLOAT)
    tc = SingleLogTestCase(
        typed_value=tv,
        expected_value=None,
        expected_error=CastingError(
            error_msg="could not convert string to float: 'fruit'",
            typed_value=tv,
        ),
    )
    table_test([tc])


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_numeric_string_to_int_error():
    tv = TypedValue(value="10.2", type=ArizeTypes.INT)
    tc = SingleLogTestCase(
        typed_value=tv,
        expected_value=None,
        expected_error=CastingError(
            error_msg="invalid literal for int() with base 10: '10.2'",
            typed_value=tv,
        ),
    )
    table_test([tc])


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_empty_string_to_numeric_error():
    tv = TypedValue(value="", type=ArizeTypes.FLOAT)
    tc = SingleLogTestCase(
        typed_value=tv,
        expected_value=None,
        expected_error=CastingError(
            error_msg="could not convert string to float: ''",
            typed_value=tv,
        ),
    )
    table_test([tc])


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_float_to_int_error():
    tv = TypedValue(value=4.4, type=ArizeTypes.INT)
    tc = SingleLogTestCase(
        typed_value=tv,
        expected_value=None,
        expected_error=CastingError(
            # this is our custom error;
            # native python float->int casting succeeds by taking the floor of the float.
            error_msg="Cannot convert float with non-zero fractional part to int",
            typed_value=tv,
        ),
    )
    table_test([tc])


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_cast_to_float_no_error():
    tc1 = SingleLogTestCase(
        typed_value=TypedValue(value=1, type=ArizeTypes.FLOAT),
        expected_value=1.0,
    )
    tc2 = SingleLogTestCase(
        typed_value=TypedValue(value="7.7", type=ArizeTypes.FLOAT),
        expected_value=7.7,
    )
    tc3 = SingleLogTestCase(
        typed_value=TypedValue(value="NaN", type=ArizeTypes.FLOAT),
        expected_value=float("NaN"),
    )
    table_test([tc1, tc2, tc3])


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_cast_to_int_no_error():
    tc1 = SingleLogTestCase(
        typed_value=TypedValue(value=1.0, type=ArizeTypes.INT), expected_value=1
    )
    tc2 = SingleLogTestCase(
        typed_value=TypedValue(value="7", type=ArizeTypes.INT), expected_value=7
    )
    tc3 = SingleLogTestCase(
        typed_value=TypedValue(value=None, type=ArizeTypes.INT),
        expected_value=None,
    )
    table_test([tc1, tc2, tc3])


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_cast_to_string_no_error():
    tc1 = SingleLogTestCase(
        typed_value=TypedValue(value=1.0, type=ArizeTypes.STR),
        expected_value="1.0",
    )
    tc2 = SingleLogTestCase(
        typed_value=TypedValue(value=float("NaN"), type=ArizeTypes.STR),
        expected_value=None,
    )
    tc3 = SingleLogTestCase(
        typed_value=TypedValue(value=None, type=ArizeTypes.STR),
        expected_value=None,
    )
    table_test([tc1, tc2, tc3])


def table_test(test_cases: List[SingleLogTestCase]):
    for test_case in test_cases:
        try:
            v = cast_value(test_case.typed_value)
        except Exception as e:
            if test_case.expected_error is None:
                pytest.fail("Unexpected error!")
            else:
                assert isinstance(e, CastingError)
                assert e.typed_value == test_case.expected_error.typed_value
                assert e.error_msg == test_case.expected_error.error_msg
        else:
            if test_case.expected_value is None:
                assert v is None
            elif not isinstance(test_case.expected_value, str) and math.isnan(
                test_case.expected_value
            ):
                assert math.isnan(v)
            else:
                assert test_case.expected_value == v


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
