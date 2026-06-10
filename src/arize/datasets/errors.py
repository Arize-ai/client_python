"""Dataset-specific exception classes."""

from abc import ABC, abstractmethod


class DatasetError(Exception, ABC):
    """Base exception for dataset-related errors."""

    def __str__(self) -> str:
        """Return a human-readable error message."""
        return self.error_message()

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""

    @abstractmethod
    def error_message(self) -> str:
        """Return the error message for this exception."""


class InvalidSessionError(DatasetError):
    """Raised when credentials are not provided or invalid."""

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "Credentials not provided or invalid. Please pass in the correct api_key when "
            "initiating a new ArizeExportClient. Alternatively, you can set up credentials "
            "in a profile or as an environment variable"
        )

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "InvalidSessionError()"


class InvalidConfigFileError(DatasetError):
    """Raised when configuration file is invalid or misconfigured."""

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return "Invalid/Misconfigured Configuration File"

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "InvalidConfigFileError()"


class IDColumnUniqueConstraintError(DatasetError):
    """Raised when id column contains duplicate values."""

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return "'id' column must contain unique values"

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "IDColumnUniqueConstraintError()"


class RequiredColumnsError(DatasetError):
    """Raised when required columns are missing from the dataset."""

    def __init__(self, missing_columns: set) -> None:
        """Initialize the exception with missing columns context.

        Args:
            missing_columns: Set of required columns that are missing.
        """
        self.missing_columns = missing_columns

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return f"Missing required columns: {self.missing_columns}"

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return f"RequiredColumnsError({self.missing_columns})"


class EmptyDatasetError(DatasetError):
    """Raised when dataset :class:`pandas.DataFrame` has no rows."""

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return "DataFrame must have at least one row in it."

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "EmptyDatasetError()"


class BinaryColumnError(DatasetError):
    """Raised when one or more columns contain Python bytes values.

    Dataset columns must be text or numeric. Python ``bytes`` values are
    encoded as a binary column type that the dataset service cannot store as
    text, so they are rejected before upload. Pass text as :class:`str`
    instead. All offending columns are reported together so they can be fixed
    in one pass.
    """

    def __init__(self, column_names: list[str]) -> None:
        """Initialize the exception with the offending column names.

        Args:
            column_names: Names of the columns whose values are bytes.
        """
        self.column_names = column_names

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return f"these columns contain bytes; pass text as str: {self.column_names}"

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return f"BinaryColumnError({self.column_names!r})"
