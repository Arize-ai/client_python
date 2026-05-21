"""Space-related exception classes."""


class AmbiguousNameError(Exception):
    """Raised when a space name matches multiple spaces.

    This can happen when the same name exists across different organizations.
    Use the space ID instead of the name to disambiguate.
    """

    def __init__(
        self,
        resource_type: str,
        name: str,
        matching_ids: list[str],
    ) -> None:
        """Initialize with the resource type, name, and conflicting IDs.

        Args:
            resource_type: Human-readable resource kind (e.g. ``"space"``).
            name: The ambiguous name that matched multiple resources.
            matching_ids: IDs of the resources sharing the name.
        """
        ids_str = ", ".join(matching_ids)
        super().__init__(
            f"Multiple {resource_type}s named '{name}' found. "
            f"Use a {resource_type} ID to disambiguate. "
            f"Matching IDs: {ids_str}"
        )
        self.resource_type = resource_type
        self.resource_name = name
        self.matching_ids = matching_ids


class MissingSpaceIDError(Exception):
    """Raised when space ID is required but not provided."""

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
        return "Missing Space ID: pass space_id explicitly"
