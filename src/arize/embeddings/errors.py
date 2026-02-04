"""Embedding generation exception classes."""


class InvalidIndexError(Exception):
    """Raised when :class:`pandas.DataFrame` or Series has an invalid index."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Index_Error"

    def __str__(self) -> str:
        """Return a human-readable error message."""
        return self.error_message()

    def __init__(self, field_name: str) -> None:
        """Initialize the exception with field name context.

        Args:
            field_name: Name of the :class:`pandas.DataFrame` or Series field with invalid index.
        """
        self.field_name = field_name

    def error_message(self) -> str:
        """Return the error message for this exception."""
        if self.field_name == "DataFrame":
            return (
                f"The index of the {self.field_name} is invalid; "
                f"reset the index by using df.reset_index(drop=True, inplace=True)"
            )
        return (
            f"The index of the Series given by the column '{self.field_name}' is invalid; "
            f"reset the index by using df.reset_index(drop=True, inplace=True)"
        )


class HuggingFaceRepositoryNotFound(Exception):
    """Raised when HuggingFace model repository is not found."""

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "HuggingFace_Repository_Not_Found_Error"

    def __str__(self) -> str:
        """Return a human-readable error message."""
        return self.error_message()

    def __init__(self, model_name: str) -> None:
        """Initialize the exception with model name context.

        Args:
            model_name: Name of the HuggingFace model that was not found.
        """
        self.model_name = model_name

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return (
            f"The given model name '{self.model_name}' is not a valid model identifier listed on "
            "'https://huggingface.co/models'. "
            "If this is a private repository, log in with `huggingface-cli login` or importing "
            "`login` from `huggingface_hub` if you are using a notebook. "
            "Learn more in https://huggingface.co/docs/huggingface_hub/quick-start#login"
        )
