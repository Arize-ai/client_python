"""Custom exceptions and error types for the Arize SDK."""


class MissingResponseFieldError(Exception):
    """Raised when a server response omits a value needed by an operation."""

    def __init__(self, field: str) -> None:
        """Store the missing field and describe the failed operation."""
        self.field = field
        super().__init__(
            f"Server response omits a field required by this operation: {field}"
        )
