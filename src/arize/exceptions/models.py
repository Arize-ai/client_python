"""Model and project-related exception classes."""


class MissingProjectNameError(Exception):
    """Raised when project name is required but not provided."""

    def __init__(self, message: str = "") -> None:
        """Initialize the exception with an optional custom message.

        Args:
            message: Custom error message, or empty string for default.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the error message."""
        return self.message or self._default_message()

    @staticmethod
    def _default_message() -> str:
        return "Missing Project Name: pass project_name explicitly"


class MissingModelNameError(Exception):
    """Raised when model name is required but not provided."""

    def __init__(self, message: str = "") -> None:
        """Initialize the exception with an optional custom message.

        Args:
            message: Custom error message, or empty string for default.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the error message."""
        return self.message or self._default_message()

    @staticmethod
    def _default_message() -> str:
        return "Missing Model Name: pass model name explicitly"
