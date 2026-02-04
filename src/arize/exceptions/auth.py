"""Authentication-related exception classes."""

from arize.constants.config import ENV_API_KEY


class MissingAPIKeyError(Exception):
    """Raised when API key is not provided via environment or configuration."""

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
        return f"Missing API key: Set '{ENV_API_KEY}' environment variable or pass api_key explicitly"
