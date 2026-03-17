"""HTTP response exception classes."""


class APIError(Exception):
    """Raised when the server returns a non-2xx HTTP response.

    For 401/403 responses see :class:`arize.exceptions.auth.AuthenticationError`.
    """

    def __init__(self, status_code: int, message: str = "") -> None:
        """Initialize the exception with the HTTP status code and optional message.

        Args:
            status_code: The HTTP status code returned by the server.
            message: Server-provided error message, if any.
        """
        self.status_code = status_code
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        """Return the error message."""
        suffix = f": {self.message}" if self.message else ""
        return f"Request failed (HTTP {self.status_code}){suffix}"
