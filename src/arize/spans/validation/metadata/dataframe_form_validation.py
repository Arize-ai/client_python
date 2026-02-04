"""DataFrame form validation for span metadata."""

import pandas as pd

from arize.exceptions.base import ValidationError
from arize.spans.columns import SPAN_SPAN_ID_COL


class MetadataFormError(ValidationError):
    """Raised when metadata :class:`pandas.DataFrame` structure or format is invalid."""

    def __init__(self, message: str, resolution: str) -> None:
        """Initialize the exception with metadata form error context.

        Args:
            message: Error message describing the invalid DataFrame structure.
            resolution: Guidance on how to resolve the error.
        """
        self.message = message
        self.resolution = resolution

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Metadata_Form_Error"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return f"{self.message} {self.resolution}"


def validate_dataframe_form(
    metadata_dataframe: pd.DataFrame,
    patch_document_column_name: str = "patch_document",
) -> list[ValidationError]:
    """Validates the structure of the metadata update dataframe.

    Args:
        metadata_dataframe: DataFrame with span IDs and patch documents or attributes.metadata.* columns
        patch_document_column_name: Name of the column containing patch documents

    Returns:
        A list of validation errors, empty if none found
    """
    errors: list[ValidationError] = []

    # Check for empty dataframe
    if metadata_dataframe.empty:
        errors.append(
            MetadataFormError(
                "metadata_dataframe is empty",
                "The metadata_dataframe is empty. No data to send.",
            )
        )
        return errors

    # Check for required span_id column
    if SPAN_SPAN_ID_COL.name not in metadata_dataframe.columns:
        errors.append(
            MetadataFormError(
                f"Missing required column: {SPAN_SPAN_ID_COL.name}",
                f"The metadata_dataframe must contain the span ID column: {SPAN_SPAN_ID_COL.name}.",
            )
        )
        return errors

    # Check for metadata columns - either patch_document or attributes.metadata.* columns
    has_patch_document = (
        patch_document_column_name in metadata_dataframe.columns
    )
    metadata_prefix = "attributes.metadata."
    metadata_columns = [
        col
        for col in metadata_dataframe.columns
        if col.startswith(metadata_prefix)
    ]
    has_metadata_fields = len(metadata_columns) > 0

    if not has_patch_document and not has_metadata_fields:
        errors.append(
            MetadataFormError(
                "Missing metadata columns",
                f"The metadata_dataframe must contain either the patch document column "
                f"'{patch_document_column_name}' or at least one column with the prefix "
                f"'{metadata_prefix}'.",
            )
        )
        return errors

    # Check for null values in required columns
    null_columns = []

    # Span ID cannot be null
    if metadata_dataframe[SPAN_SPAN_ID_COL.name].isna().any():
        null_columns.append(SPAN_SPAN_ID_COL.name)

    # If using patch_document, it cannot be null
    if (
        has_patch_document
        and metadata_dataframe[patch_document_column_name].isna().any()
    ):
        null_columns.append(patch_document_column_name)

    # If using metadata fields, check each one
    if has_metadata_fields:
        null_columns.extend(
            [
                col
                for col in metadata_columns
                if metadata_dataframe[col]
                .isna()
                .all()  # All values in column are null
            ]
        )

    if null_columns:
        errors.append(
            MetadataFormError(
                f"Columns with null values: {', '.join(null_columns)}",
                f"The following columns cannot contain null values: {', '.join(null_columns)}.",
            )
        )

    return errors
