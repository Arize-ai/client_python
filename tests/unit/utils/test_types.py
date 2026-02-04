"""Tests for common type definitions and validation utilities."""

import numpy as np
import pytest

from arize.utils.types import (
    is_array_of,
    is_dict_of,
    is_iterable_of,
    is_json_str,
    is_list_of,
)


@pytest.mark.unit
class TestIsJsonStr:
    """Test is_json_str function."""

    @pytest.mark.parametrize(
        "json_string",
        [
            '{"key": "value"}',
            '{"nested": {"key": 123}}',
            '["list", "of", "strings"]',
            "[1, 2, 3]",
            "null",
            "true",
            "false",
            "123",
            '"string"',
            "{}",
            "[]",
        ],
    )
    def test_valid_json_strings(self, json_string: str) -> None:
        """Should return True for valid JSON strings."""
        assert is_json_str(json_string) is True

    @pytest.mark.parametrize(
        "invalid_string",
        [
            "not json",
            "{invalid: json}",
            "{'single': 'quotes'}",
            "{key: value}",
            "[1, 2, 3,]",  # Trailing comma
            "",  # Empty string - not valid JSON
            "undefined",
            # Note: "NaN" is actually valid JSON in Python's json module
        ],
    )
    def test_invalid_json_strings(self, invalid_string: str) -> None:
        """Should return False for invalid JSON strings."""
        assert is_json_str(invalid_string) is False

    def test_type_error_handling(self) -> None:
        """Should return False when TypeError occurs."""
        assert is_json_str(None) is False  # type: ignore


@pytest.mark.unit
class TestIsArrayOf:
    """Test is_array_of function."""

    def test_numpy_array_of_ints(self) -> None:
        """Should return True for numpy array of integers."""
        arr = np.array([1, 2, 3])
        # Note: numpy integers are np.int64, not Python int
        assert is_array_of(arr, (int, np.integer)) is True

    def test_numpy_array_of_floats(self) -> None:
        """Should return True for numpy array of floats."""
        arr = np.array([1.0, 2.0, 3.0])
        # Note: numpy scalars are np.float64, not float
        assert is_array_of(arr, (float, np.floating)) is True

    def test_numpy_array_of_strings(self) -> None:
        """Should return True for numpy array of strings."""
        arr = np.array(["a", "b", "c"])
        assert is_array_of(arr, str) is True

    def test_empty_numpy_array(self) -> None:
        """Should return True for empty numpy array."""
        arr = np.array([])
        assert is_array_of(arr, int) is True

    def test_list_not_array(self) -> None:
        """Should return False for list (not numpy array)."""
        lst = [1, 2, 3]
        assert is_array_of(lst, int) is False

    def test_mixed_types_in_array(self) -> None:
        """Should return False for array with mixed types."""
        arr = np.array([1, "two", 3])
        assert is_array_of(arr, int) is False


@pytest.mark.unit
class TestIsListOf:
    """Test is_list_of function."""

    @pytest.mark.parametrize(
        "input_list,element_type,expected",
        [
            ([1, 2, 3], int, True),
            (["a", "b", "c"], str, True),
            ([1.0, 2.0, 3.0], float, True),
            ([True, False, True], bool, True),
            ([], int, True),  # Empty list
            ([1, "two", 3], int, False),  # Mixed types
            ([1, 2, None], int, False),  # Contains None
            ("string", str, False),  # Not a list
            (None, int, False),  # Not a list
        ],
    )
    def test_list_type_validation(
        self, input_list: object, element_type: type, expected: bool
    ) -> None:
        """Should correctly validate list element types."""
        assert is_list_of(input_list, element_type) is expected

    def test_empty_list_with_any_type(self) -> None:
        """Should return True for empty list with any type."""
        assert is_list_of([], int) is True
        assert is_list_of([], str) is True

    def test_tuple_not_list(self) -> None:
        """Should return False for tuple (not list)."""
        assert is_list_of((1, 2, 3), int) is False

    def test_numpy_array_not_list(self) -> None:
        """Should return False for numpy array (not list)."""
        arr = np.array([1, 2, 3])
        assert is_list_of(arr, int) is False


@pytest.mark.unit
class TestIsIterableOf:
    """Test is_iterable_of function."""

    @pytest.mark.parametrize(
        "input_val,element_type,expected",
        [
            ([1, 2, 3], int, True),
            ((1, 2, 3), int, True),
            ({1, 2, 3}, int, True),
            (["a", "b"], str, True),
            ([], int, True),
            ([1, "two"], int, False),
            ("string", str, True),  # String is iterable of chars
        ],
    )
    def test_iterable_type_validation(
        self, input_val: object, element_type: type, expected: bool
    ) -> None:
        """Should correctly validate iterable element types."""
        assert is_iterable_of(input_val, element_type) is expected  # type: ignore

    def test_numpy_array_is_iterable(self) -> None:
        """Should return True for numpy array (is iterable)."""
        arr = np.array([1, 2, 3])
        # Note: numpy integers are np.int64, not int
        assert is_iterable_of(arr, (int, np.integer)) is True

    def test_empty_iterable(self) -> None:
        """Should return True for empty iterables."""
        assert is_iterable_of([], int) is True
        assert is_iterable_of((), str) is True


@pytest.mark.unit
class TestIsDictOf:
    """Test is_dict_of function."""

    def test_dict_with_str_keys_and_int_values(self) -> None:
        """Should return True for dict with correct key and value types."""
        d = {"a": 1, "b": 2, "c": 3}
        assert is_dict_of(d, str, int) is True

    def test_dict_with_str_keys_and_str_values(self) -> None:
        """Should return True for dict with matching types."""
        d = {"key1": "value1", "key2": "value2"}
        assert is_dict_of(d, str, str) is True

    def test_empty_dict(self) -> None:
        """Should return True for empty dict."""
        assert is_dict_of({}, str) is True
        assert is_dict_of({}, str, int) is True

    def test_dict_with_wrong_key_type(self) -> None:
        """Should return False when key types don't match."""
        d = {1: "value1", 2: "value2"}
        assert is_dict_of(d, str, str) is False

    def test_dict_with_wrong_value_type(self) -> None:
        """Should return False when value types don't match."""
        d = {"a": 1, "b": 2}
        assert is_dict_of(d, str, str) is False

    def test_dict_with_list_values(self) -> None:
        """Should return True for dict with list values of correct type."""
        d = {"a": [1, 2, 3], "b": [4, 5, 6]}
        assert is_dict_of(d, str, value_list_allowed_types=int) is True

    def test_dict_with_mixed_values_and_list_values(self) -> None:
        """Should return True for dict with mixed single and list values."""
        d = {"a": 1, "b": [2, 3], "c": 4}
        assert is_dict_of(d, str, int, int) is True

    def test_dict_with_tuple_key_types(self) -> None:
        """Should support tuple of allowed key types."""
        d = {"a": 1, 2: 3}
        assert is_dict_of(d, (str, int), int) is True

    def test_dict_with_tuple_value_types(self) -> None:
        """Should support tuple of allowed value types."""
        d = {"a": 1, "b": "string", "c": 2}
        assert is_dict_of(d, str, (int, str)) is True

    def test_not_a_dict(self) -> None:
        """Should return False for non-dict values."""
        assert is_dict_of([1, 2, 3], str, int) is False
        assert is_dict_of("string", str, str) is False
        assert is_dict_of(None, str, int) is False

    def test_dict_with_list_values_wrong_list_type(self) -> None:
        """Should return False when list values have wrong types."""
        d = {"a": ["x", "y"], "b": [1, 2]}
        assert is_dict_of(d, str, value_list_allowed_types=int) is False

    def test_dict_keys_only_validation(self) -> None:
        """Should validate only keys when no value types specified."""
        d = {"a": 1, "b": "string", "c": [1, 2]}
        assert is_dict_of(d, str) is True

    def test_dict_with_tuple_value_list_allowed_types(self) -> None:
        """Should handle tuple for value_list_allowed_types."""
        d = {"a": [1, 2], "b": ["x", "y"]}
        assert is_dict_of(d, str, value_list_allowed_types=(int, str)) is True
