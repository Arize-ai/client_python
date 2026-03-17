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


class AuthenticationError(Exception):
    """Raised when the server rejects the request due to invalid credentials (HTTP 401/403)."""

    def __init__(self, status_code: int, message: str = "") -> None:
        """Initialize the exception with the HTTP status code and optional message.

        Args:
            status_code: The HTTP status code returned by the server (e.g. 401, 403).
            message: Server-provided error message, if any.
        """
        self.status_code = status_code
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        """Return the error message."""
        suffix = f": {self.message}" if self.message else ""
        return (
            f"Authentication failed (HTTP {self.status_code}){suffix}. "
            "Verify your API key and space ID are correct."
        )
