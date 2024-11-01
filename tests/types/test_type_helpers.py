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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
