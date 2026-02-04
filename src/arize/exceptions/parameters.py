"""Parameter validation exception classes."""

from arize.constants.ml import MAX_NUMBER_OF_EMBEDDINGS
from arize.exceptions.base import ValidationError


class InvalidModelVersion(ValidationError):
    """Raised when model version is empty or invalid."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Model_Version"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return "Model version must be a nonempty string."


class InvalidProjectName(ValidationError):
    """Raised when project name is empty or invalid."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Project_Name"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            "Project Name must be a nonempty string. "
            "If Model ID was used instead of Project Name, "
            "it must be a nonempty string."
        )


class InvalidNumberOfEmbeddings(ValidationError):
    """Raised when number of embedding features exceeds the maximum allowed."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Number_Of_Embeddings"

    def __init__(self, number_of_embeddings: int) -> None:
        """Initialize the exception with embedding count context.

        Args:
            number_of_embeddings: The number of embeddings found in the schema.
        """
        self.number_of_embeddings = number_of_embeddings

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"The schema contains {self.number_of_embeddings} different embeddings when a maximum of "
            f"{MAX_NUMBER_OF_EMBEDDINGS} is allowed."
        )


class InvalidValueType(Exception):
    """Raised when a value has an invalid or unexpected type."""

    def __init__(
        self,
        value_name: str,
        value: object,
        correct_type: str,
    ) -> None:
        """Initialize the exception with value type validation context.

        Args:
            value_name: Name of the value with invalid type.
            value: The actual value that has the wrong type.
            correct_type: Description of the expected type.
        """
        self.value_name = value_name
        self.value = value
        self.correct_type = correct_type

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Value_Type"

    def __str__(self) -> str:
        """Return a human-readable error message."""
        return self.error_message()

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"{self.value_name} with value {self.value} is of type {type(self.value).__name__}, "
            f"but expected {self.correct_type}"
        )
