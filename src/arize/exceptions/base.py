"""Base exception classes and common error messages."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

INVALID_ARROW_CONVERSION_MSG = (
    "The dataframe needs to convert to pyarrow but has failed to do so. "
    "There may be unrecognized data types in the dataframe. "
    "Another reason may be that a column in the dataframe has a mix of strings and "
    "numbers, in which case you may want to convert the strings in that column to NaN. "
)


class ValidationError(Exception, ABC):
    """Base exception for validation errors in data and schema validation."""

    def __str__(self) -> str:
        """Return a human-readable error message."""
        return self.error_message()

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""

    @abstractmethod
    def error_message(self) -> str:
        """Return the error message for this exception."""


class ValidationFailure(Exception):
    """Raised when one or more validation errors occur during validation."""

    def __init__(self, errors: list[ValidationError]) -> None:
        """Initialize the exception with a list of validation errors.

        Args:
            errors: List of ValidationError instances that occurred.
        """
        self.errors = errors


class InvalidFieldTypeConversion(ValidationError):
    """Raised when fields cannot be converted to required type."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Input_Type_Conversion"

    def __init__(self, fields: Iterable, type: str) -> None:
        """Initialize the exception with type conversion context.

        Args:
            fields: Fields that failed type conversion.
            type: Target type that fields should be convertible to.
        """
        self.fields = fields
        self.type = type

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"The following fields must be convertible to {self.type}: "
            f"{', '.join(map(str, self.fields))}."
        )


class InvalidDataFrameIndex(ValidationError):
    """Raised when DataFrame has an invalid index that needs to be reset."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Index"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "The index of the dataframe is invalid; "
            "reset the index by using df.reset_index(drop=True, inplace=True)"
        )
