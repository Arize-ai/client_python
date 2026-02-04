"""Unit tests for arize._exporter.validation module."""

from __future__ import annotations

from datetime import datetime

import pytest

from arize._exporter.validation import (
    validate_input_type,
    validate_input_value,
    validate_start_end_time,
)


@pytest.mark.unit
class TestValidateInputType:
    """Test validate_input_type function."""

    def test_valid_type_passes(self) -> None:
        """Test that validation passes for correct type."""
        validate_input_type("test", "param", str)
        validate_input_type(123, "param", int)
        validate_input_type([1, 2], "param", list)

    def test_wrong_type_raises_type_error(self) -> None:
        """Test that wrong type raises TypeError."""
        with pytest.raises(
            TypeError, match=r"param 123 is type .*, but must be a str"
        ):
            validate_input_type(123, "param", str)

    def test_none_not_allowed_raises_type_error(self) -> None:
        """Test that None raises TypeError when not allowed."""
        with pytest.raises(
            TypeError, match=r"param None is type .*, but must not be None"
        ):
            validate_input_type(None, "param", str, allow_none=False)

    def test_none_allowed_passes(self) -> None:
        """Test that None passes when allow_none=True."""
        validate_input_type(None, "param", str, allow_none=True)

    def test_subclass_instance_passes(self) -> None:
        """Test that subclass instances pass isinstance check."""

        class CustomList(list):
            pass

        custom = CustomList([1, 2, 3])
        validate_input_type(custom, "param", list)

    def test_error_message_format(self) -> None:
        """Test that error message contains parameter name and type."""
        with pytest.raises(
            TypeError,
            match=r"my_param \[1, 2, 3\] is type <class 'list'>, but must be a str",
        ):
            validate_input_type([1, 2, 3], "my_param", str)

    def test_multiple_types_with_union(self) -> None:
        """Test validation with Union types."""
        validate_input_type("test", "param", (str, int))
        validate_input_type(123, "param", (str, int))

    def test_generic_types_list_dict(self) -> None:
        """Test with generic collection types."""
        validate_input_type({"a": 1}, "param", dict)
        validate_input_type([1, 2, 3], "param", list)
        validate_input_type((1, 2, 3), "param", tuple)


@pytest.mark.unit
class TestValidateInputValue:
    """Test validate_input_value function."""

    def test_valid_value_in_choices(self) -> None:
        """Test that validation passes when value is in choices."""
        validate_input_value(
            "option1", "param", ("option1", "option2", "option3")
        )
        validate_input_value(1, "param", (1, 2, 3))

    def test_invalid_value_raises_value_error(self) -> None:
        """Test that invalid value raises ValueError."""
        with pytest.raises(
            ValueError, match="param is option4, but must be one of"
        ):
            validate_input_value(
                "option4", "param", ("option1", "option2", "option3")
            )

    def test_error_message_format(self) -> None:
        """Test that error message contains all choices."""
        with pytest.raises(
            ValueError,
            match=r"my_param is invalid, but must be one of choice1, choice2, choice3",
        ):
            validate_input_value(
                "invalid", "my_param", ("choice1", "choice2", "choice3")
            )

    def test_empty_choices_tuple(self) -> None:
        """Test with empty choices tuple."""
        with pytest.raises(ValueError):
            validate_input_value("any", "param", ())


@pytest.mark.unit
class TestValidateStartEndTime:
    """Test validate_start_end_time function."""

    def test_start_before_end_passes(self) -> None:
        """Test that validation passes when start is before end."""
        start = datetime(2024, 1, 1, 10, 0)
        end = datetime(2024, 1, 1, 12, 0)
        validate_start_end_time(start, end)

    def test_start_equals_end_raises_value_error(self) -> None:
        """Test that equal start and end times raise ValueError."""
        time = datetime(2024, 1, 1, 10, 0)
        with pytest.raises(
            ValueError, match="start_time must be before end_time"
        ):
            validate_start_end_time(time, time)

    def test_start_after_end_raises_value_error(self) -> None:
        """Test that start after end raises ValueError."""
        start = datetime(2024, 1, 1, 12, 0)
        end = datetime(2024, 1, 1, 10, 0)
        with pytest.raises(
            ValueError, match="start_time must be before end_time"
        ):
            validate_start_end_time(start, end)
