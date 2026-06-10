"""Configuration validation exceptions."""

from __future__ import annotations


class MultipleEndpointOverridesError(Exception):
    """Raised when multiple endpoint override options are provided.

    Only one of the following can be specified: region, single_host/single_port, or base_domain.
    """

    def __init__(self, message: str) -> None:
        """Initialize the exception with an optional custom message.

        Args:
            message: Custom error message, or empty string.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the error message."""
        return self.message


class InvalidDefaultHeadersError(Exception):
    """Raised when ``default_headers`` contains an invalid or reserved header.

    Covers all default-header validation failures: non-string keys/values,
    empty keys, control-character/injection attempts, ``Grpc-Metadata-``
    prefixed keys (reserved for SDK-owned gRPC translation), and keys that
    collide with the SDK's built-in headers.
    """

    def __init__(self, message: str) -> None:
        """Initialize the exception with a custom message.

        Args:
            message: Custom error message describing the offending header.
        """
        self.message = message

    def __str__(self) -> str:
        """Return the error message."""
        return self.message
