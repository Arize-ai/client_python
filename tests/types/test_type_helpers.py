import pytest

from arize.utils.types import is_dict_of, is_list_of


def test_is_dict_of():
    # Assert only key
    assert (
        is_dict_of(
            {"class1": 0.1, "class2": 0.2},
            key_allowed_types=str,
        )
        is True
    )
    # Assert key and value exact types
    assert (
        is_dict_of(
            {"class1": 0.1, "class2": 0.2},
            key_allowed_types=str,
            value_allowed_types=float,
        )
        is True
    )
    assert (
        is_dict_of(
            {"class1": 0.1, "class2": 0.2},
            key_allowed_types=str,
            value_allowed_types=int,
        )
        is False
    )
    # Assert key and value union types
    assert (
        is_dict_of(
            {"class1": 0.1, "class2": 0.2},
            key_allowed_types=str,
            value_allowed_types=(str, float),
        )
        is True
    )
    # Assert key and exact list of value types
    assert (
        is_dict_of(
            {"class1": [1, 2], "class2": [3, 4]},
            key_allowed_types=str,
            value_list_allowed_types=int,
        )
        is True
    )
    # Assert key and exact list of value types
    assert (
        is_dict_of(
            {"class1": [1, 2], "class2": [3, 4]},
            key_allowed_types=str,
            value_list_allowed_types=str,
        )
        is False
    )
    # Assert key and union list of value types
    assert (
        is_dict_of(
            {"class1": [1, 2], "class2": [3, 4]},
            key_allowed_types=str,
            value_list_allowed_types=(str, int),
        )
        is True
    )
    assert (
        is_dict_of(
            {"class1": [1, 2], "class2": ["a", "b"]},
            key_allowed_types=str,
            value_list_allowed_types=(str, int),
        )
        is True
    )
    # Assert key and value and list of value types
    assert (
        is_dict_of(
            {"class1": 1, "class2": ["a", "b"], "class3": [0.4, 0.7]},
            key_allowed_types=str,
            value_allowed_types=int,
            value_list_allowed_types=(str, float),
        )
        is True
    )
    assert (
        is_dict_of(
            {"class1": 1, "class2": ["a", "b"], "class3": [0.4, 0.7]},
            key_allowed_types=str,
            value_allowed_types=str,
            value_list_allowed_types=(str, float),
        )
        is False
    )


def test_is_list_of():
    assert is_list_of([1, 2], int) is True
    assert is_list_of([1, 2], float) is False
    assert is_list_of(["x", 2], int) is False
    assert is_list_of(["x", 2], (str, int)) is True


def test_is_dict_of_edge_cases():
    # Test empty dictionary
    assert is_dict_of({}, key_allowed_types=str) is True, (
        "Empty dictionary should return True"
    )
    # Test None as input
    assert is_dict_of(None, key_allowed_types=str) is False, (
        "None input should return False"
    )
    # Test dictionary with a None value
    assert (
        is_dict_of(
            {"key": None},
            key_allowed_types=str,
            value_allowed_types=(str, type(None)),
        )
        is True
    ), "Dictionary with None value should pass if None is an allowed type"
    # Test nested dictionary by specifying allowed value types that do not include dict
    assert (
        is_dict_of(
            {"key": {"nested_key": "value"}},
            key_allowed_types=str,
            value_allowed_types=(str, int),
        )
        is False
    ), (
        "Nested dictionaries should return False if dict is not an allowed value type"
    )
    # Test nested dictionary when no value types are specified (documents current behavior)
    assert (
        is_dict_of({"key": {"nested_key": "value"}}, key_allowed_types=str)
        is True
    ), "Nested dictionaries are allowed when value types are not specified"


def test_is_list_of_edge_cases():
    # Test empty list
    assert is_list_of([], str) is True, "Empty list should return True"
    # Test None as input
    assert is_list_of(None, str) is False, "None input should return False"
    # Test list with a None value
    assert is_list_of([None], str) is False, (
        "List with None value should return False if None is not an allowed type"
    )
    # Test list with a None value where None is an allowed type
    assert is_list_of([None], (str, type(None))) is True, (
        "List with None value should pass if None is an allowed type"
    )
    # Test nested list
    assert is_list_of([["a", "b"]], str) is False, (
        "Nested lists should return False"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
