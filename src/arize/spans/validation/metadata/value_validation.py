"""Value validation logic for span metadata."""

import json

import pandas as pd

from arize.constants.spans import (
    MAX_JSON_NESTING_DEPTH,
    MAX_JSON_STRING_SIZE,
)
from arize.exceptions.base import ValidationError
from arize.spans.columns import SPAN_SPAN_ID_COL


class MetadataValueError(ValidationError):
    """Raised when metadata values are invalid or violate constraints."""

    def __init__(self, message: str, resolution: str) -> None:
        """Initialize the exception with metadata value error context.

        Args:
            message: Error message describing the invalid value.
            resolution: Guidance on how to resolve the error.
        """
        self.message = message
        self.resolution = resolution

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Metadata_Value_Error"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return f"{self.message} {self.resolution}"


class InvalidPatchDocumentFormat(ValidationError):
    """Raised when patch document format is invalid or cannot be parsed."""

    def __init__(self, row_idx: int, message: str) -> None:
        """Initialize the exception with patch document format error context.

        Args:
            row_idx: The row index where the invalid patch was found.
            message: Detailed error message describing the format issue.
        """
        self.row_idx = row_idx
        self.message = message

    def __repr__(self) -> str:
        """Return a string representation for debugging and logging."""
        return "Invalid_Patch_Document_Format"

    def error_message(self) -> str:
        """Return the error message for this exception."""
        return f"Row {self.row_idx}: {self.message}"


def calculate_json_depth(obj: object, current_depth: int = 1) -> int:
    """Calculate the maximum nesting depth of a JSON object.

    Stops recursing once MAX_JSON_NESTING_DEPTH + 1 is reached for efficiency.
    """
    # If we've already exceeded the max depth, return the current depth to avoid unnecessary recursion
    if current_depth > MAX_JSON_NESTING_DEPTH:
        return current_depth

    if isinstance(obj, dict) and obj:
        return max(
            [calculate_json_depth(v, current_depth + 1) for v in obj.values()]
        )
    if isinstance(obj, list) and obj:
        return max(
            [calculate_json_depth(item, current_depth + 1) for item in obj]
        )
    return current_depth


def validate_values(
    metadata_dataframe: pd.DataFrame,
    patch_document_column_name: str = "patch_document",
) -> list[ValidationError]:
    """Validates the values in the metadata update dataframe.

    Args:
        metadata_dataframe: DataFrame with span IDs and patch documents or attributes.metadata.* columns
        patch_document_column_name: Name of the column containing patch documents

    Returns:
        A list of validation errors, empty if none found
    """
    errors: list[ValidationError] = []

    # Skip validation if span_id column is not present
    if SPAN_SPAN_ID_COL.name not in metadata_dataframe.columns:
        return errors

    # Validate span_id format - ensure they're strings and not empty
    invalid_span_ids = []
    for idx, span_id in enumerate(metadata_dataframe[SPAN_SPAN_ID_COL.name]):
        if not isinstance(span_id, str) or not span_id.strip():
            invalid_span_ids.append(f"Row {idx}: '{span_id}'")

    if invalid_span_ids:
        sample = invalid_span_ids[:5]
        errors.append(
            MetadataValueError(
                f"Invalid span IDs: {', '.join(sample)}{'...' if len(invalid_span_ids) > 5 else ''}",
                "Span IDs must be non-empty strings.",
            )
        )

    # Check if using patch_document or attributes.metadata.* columns
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

    # Validate patch documents if present
    if has_patch_document:
        invalid_patches = []
        oversized_patches = []
        deep_patches = []

        for idx, patch in enumerate(
            metadata_dataframe[patch_document_column_name]
        ):
            if isinstance(patch, dict):
                # Validate size of serialized dict
                try:
                    json_str = json.dumps(patch)
                    if len(json_str) > MAX_JSON_STRING_SIZE:
                        oversized_patches.append(f"Row {idx}")

                    # Check nesting depth
                    depth = calculate_json_depth(patch)
                    if depth > MAX_JSON_NESTING_DEPTH:
                        deep_patches.append(f"Row {idx}")
                except (TypeError, OverflowError):
                    invalid_patches.append(f"Row {idx}")
            elif isinstance(patch, str):
                # First check string size
                if len(patch) > MAX_JSON_STRING_SIZE:
                    oversized_patches.append(f"Row {idx}")
                    continue

                try:
                    # Verify it's valid JSON
                    parsed = json.loads(patch)
                    if not isinstance(parsed, dict):
                        invalid_patches.append(f"Row {idx}")
                        continue

                    # Check nesting depth
                    depth = calculate_json_depth(parsed)
                    if depth > MAX_JSON_NESTING_DEPTH:
                        deep_patches.append(f"Row {idx}")
                except json.JSONDecodeError:
                    invalid_patches.append(f"Row {idx}")
            else:
                invalid_patches.append(f"Row {idx}")

        if invalid_patches:
            sample = invalid_patches[:5]
            errors.append(
                MetadataValueError(
                    f"Invalid JSON patches: {', '.join(sample)}{'...' if len(invalid_patches) > 5 else ''}",
                    "Patch documents must be valid JSON strings or dictionaries.",
                )
            )

        if oversized_patches:
            sample = oversized_patches[:5]
            errors.append(
                MetadataValueError(
                    f"Oversized JSON patches: {', '.join(sample)}"
                    f"{'...' if len(oversized_patches) > 5 else ''}",
                    f"JSON patches must not exceed {MAX_JSON_STRING_SIZE} bytes when serialized.",
                )
            )

        if deep_patches:
            sample = deep_patches[:5]
            errors.append(
                MetadataValueError(
                    f"Excessively nested JSON patches: {', '.join(sample)}"
                    f"{'...' if len(deep_patches) > 5 else ''}",
                    f"JSON patches must not exceed {MAX_JSON_NESTING_DEPTH} levels of nesting.",
                )
            )

    # Validate metadata fields if present
    if has_metadata_fields:
        invalid_fields = []
        oversized_fields = []
        deep_fields = []

        for col in metadata_columns:
            field_name = col.replace(metadata_prefix, "")
            if not field_name:
                invalid_fields.append(col)
                continue

            # Check for unsupported data types in each column
            # Primary support for: strings, integers, floats
            # Other types (booleans, lists, dicts) will be serialized
            # None values are supported and will be set to null
            for idx, value in enumerate(metadata_dataframe[col]):
                # None values are valid and will be set to null
                if value is None:
                    continue

                # Check string size directly for string values
                if isinstance(value, str) and len(value) > MAX_JSON_STRING_SIZE:
                    oversized_fields.append(f"{col} (Row {idx})")
                    continue

                # For nested structures (lists/dicts), check size and depth
                if isinstance(value, (list, dict)):
                    try:
                        # Check depth
                        depth = calculate_json_depth(value)
                        if depth > MAX_JSON_NESTING_DEPTH:
                            deep_fields.append(f"{col} (Row {idx})")
                            continue

                        # Check serialized size
                        json_str = json.dumps(value)
                        if len(json_str) > MAX_JSON_STRING_SIZE:
                            oversized_fields.append(f"{col} (Row {idx})")
                            continue
                    except (TypeError, OverflowError):
                        invalid_fields.append(f"{col} (Row {idx})")
                        continue

                # For all other types, verify they can be serialized to JSON
                # Note: booleans will be converted to strings
                if (
                    not isinstance(value, (str, int, float, bool, list, dict))
                    and value is not None
                ):
                    try:
                        # Try to serialize to see if it can be transmitted
                        json_str = json.dumps(value)
                        if len(json_str) > MAX_JSON_STRING_SIZE:
                            oversized_fields.append(f"{col} (Row {idx})")
                    except (TypeError, OverflowError):
                        invalid_fields.append(f"{col} (Row {idx})")

        if invalid_fields:
            sample = invalid_fields[:5]
            errors.append(
                MetadataValueError(
                    f"Invalid metadata fields: {', '.join(sample)}{'...' if len(invalid_fields) > 5 else ''}",
                    "Metadata fields must have valid names and be serializable. "
                    "Strings, numbers, and null values are fully supported; "
                    "other types will be serialized during transmission.",
                )
            )

        if oversized_fields:
            sample = oversized_fields[:5]
            errors.append(
                MetadataValueError(
                    f"Oversized metadata fields: {', '.join(sample)}"
                    f"{'...' if len(oversized_fields) > 5 else ''}",
                    f"Metadata field values must not exceed {MAX_JSON_STRING_SIZE} bytes when serialized.",
                )
            )

        if deep_fields:
            sample = deep_fields[:5]
            errors.append(
                MetadataValueError(
                    f"Excessively nested metadata fields: {', '.join(sample)}"
                    f"{'...' if len(deep_fields) > 5 else ''}",
                    f"Metadata field values must not exceed {MAX_JSON_NESTING_DEPTH} levels of nesting.",
                )
            )

    return errors
