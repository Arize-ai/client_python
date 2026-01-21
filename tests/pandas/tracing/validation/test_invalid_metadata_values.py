import sys
import uuid

import pandas as pd
import pytest

if sys.version_info >= (3, 8):
    from arize.pandas.tracing.columns import SPAN_SPAN_ID_COL
    from arize.pandas.tracing.validation.metadata.value_validation import (
        validate_values,
    )


def get_valid_metadata_df(num_rows=2):
    """Helper to create a valid metadata DataFrame."""
    span_ids = [str(uuid.uuid4()) for _ in range(num_rows)]

    df = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: span_ids,
            "patch_document": [
                '{"key1": "value1", "nested": {"bool_value": true}}',
                '{"key2": "value2", "tags": ["tag1", "tag2"]}',
            ][:num_rows],
        }
    )
    return df


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_valid_metadata_values():
    """Tests that valid values pass validation."""
    metadata_df = get_valid_metadata_df()

    errors = validate_values(metadata_dataframe=metadata_df)
    assert len(errors) == 0, "Expected no validation errors for valid values"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_span_id_empty():
    """Tests error for empty span_id values."""
    metadata_df = get_valid_metadata_df(1)
    metadata_df[SPAN_SPAN_ID_COL.name] = [""]  # Empty string

    errors = validate_values(metadata_dataframe=metadata_df)
    assert len(errors) > 0, "Expected error for empty span_id"
    assert any("span id" in str(error).lower() for error in errors), (
        "Expected error message about span ID"
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_span_id_non_string():
    """Tests error for non-string span_id values."""
    metadata_df = get_valid_metadata_df(1)
    # Force the type to be numeric
    metadata_df[SPAN_SPAN_ID_COL.name] = pd.Series([123], dtype=int)

    errors = validate_values(metadata_dataframe=metadata_df)
    assert len(errors) > 0, "Expected error for non-string span_id"
    assert any("span id" in str(error).lower() for error in errors), (
        "Expected error message about span ID"
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_json_syntax():
    """Tests error for invalid JSON syntax in patch_document."""
    metadata_df = get_valid_metadata_df(1)
    metadata_df["patch_document"] = ["{missing: quotes, invalid}"]

    errors = validate_values(metadata_dataframe=metadata_df)
    assert len(errors) > 0, "Expected error for invalid JSON syntax"
    assert any("json" in str(error).lower() for error in errors), (
        "Expected error message about JSON"
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_json_incomplete():
    """Tests error for incomplete JSON in patch_document."""
    metadata_df = get_valid_metadata_df(1)
    metadata_df["patch_document"] = ['{"unclosed": "object"']

    errors = validate_values(metadata_dataframe=metadata_df)
    assert len(errors) > 0, "Expected error for incomplete JSON"
    assert any("json" in str(error).lower() for error in errors), (
        "Expected error message about JSON"
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_valid_json_dict_object():
    """Tests that dictionary objects are valid for patch_document."""
    # Create DataFrame with dict objects instead of JSON strings
    metadata_df = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: [str(uuid.uuid4())],
            "patch_document": [
                {"key1": "value1", "nested": {"bool_value": True}}
            ],
        }
    )

    errors = validate_values(metadata_dataframe=metadata_df)
    assert len(errors) == 0, "Expected no errors for dict objects"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_patch_type():
    """Tests error for invalid patch_document types (not string or dict)."""
    metadata_df = get_valid_metadata_df(1)

    # Test with numeric type
    metadata_df["patch_document"] = [123]
    errors_numeric = validate_values(metadata_dataframe=metadata_df)
    assert len(errors_numeric) > 0, "Expected error for numeric patch_document"
    assert any("json" in str(error).lower() for error in errors_numeric), (
        "Expected error message about invalid type"
    )

    # Test with list type
    metadata_df["patch_document"] = [[1, 2, 3]]
    errors_list = validate_values(metadata_dataframe=metadata_df)
    assert len(errors_list) > 0, "Expected error for list-type patch_document"
    assert any("json" in str(error).lower() for error in errors_list), (
        "Expected error message about invalid type"
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_custom_patch_document_column():
    """Tests validation with custom patch_document_column_name."""
    metadata_df = get_valid_metadata_df()
    custom_name = "my_custom_patches"
    metadata_df = metadata_df.rename(columns={"patch_document": custom_name})

    # Should work with correct custom column name
    errors_valid = validate_values(
        metadata_dataframe=metadata_df, patch_document_column_name=custom_name
    )
    assert len(errors_valid) == 0, "Expected no errors with valid custom column"

    # Introduce error in custom column
    metadata_df[custom_name] = ["invalid json", "more invalid"]
    errors_invalid = validate_values(
        metadata_dataframe=metadata_df, patch_document_column_name=custom_name
    )
    assert len(errors_invalid) > 0, (
        "Expected errors with invalid custom column values"
    )
    assert any("json" in str(error).lower() for error in errors_invalid), (
        "Expected error message about JSON"
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_metadata_fields_only():
    """Tests validation with only attributes.metadata.* columns."""
    span_id = str(uuid.uuid4())
    metadata_df = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: [span_id],
            "attributes.metadata.tag": ["important"],
            "attributes.metadata.priority": ["high"],
            "attributes.metadata.numeric_value": [123],
            "attributes.metadata.bool_value": [True],
            "attributes.metadata.array_value": [["item1", "item2"]],
        }
    )

    errors = validate_values(metadata_dataframe=metadata_df)
    assert len(errors) == 0, "Expected no errors with valid metadata fields"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_metadata_fields():
    """Tests validation with invalid attributes.metadata.* values."""
    span_id = str(uuid.uuid4())

    # Create a class that cannot be serialized to JSON
    class UnserializableObject:
        def __init__(self):
            self.value = "test"

    metadata_df = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: [span_id],
            "attributes.metadata.": [
                "Invalid empty field name"
            ],  # Empty field name
            "attributes.metadata.invalid_object": [
                UnserializableObject()
            ],  # Unserializable object
        }
    )

    errors = validate_values(metadata_dataframe=metadata_df)
    assert len(errors) > 0, "Expected errors with invalid metadata fields"
    assert any(
        "invalid metadata fields" in str(error).lower() for error in errors
    ), "Expected error message about invalid metadata fields"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_combined_metadata_approach():
    """Tests validation with both attributes.metadata.* columns and patch_document."""
    span_id = str(uuid.uuid4())

    # Create DataFrame with both direct metadata fields and patch document
    metadata_df = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: [span_id],
            "attributes.metadata.visible": [True],
            "attributes.metadata.tags": [["test", "metadata", "combined"]],
            "patch_document": ['{"nested": {"value": 42, "name": "answer"}}'],
        }
    )

    # Should work with both approaches
    errors = validate_values(metadata_dataframe=metadata_df)
    assert len(errors) == 0, "Expected no errors with combined approach"

    # Now test with invalid values in both
    invalid_df = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: [span_id],
            "attributes.metadata.": [
                "Invalid empty field"
            ],  # Invalid field name
            "patch_document": ["{invalid json}"],  # Invalid JSON
        }
    )

    errors_invalid = validate_values(metadata_dataframe=invalid_df)
    assert len(errors_invalid) >= 2, (
        "Expected at least 2 errors with invalid combined approach"
    )

    error_messages = [str(error).lower() for error in errors_invalid]
    assert any("json" in msg for msg in error_messages), (
        "Expected error about invalid JSON"
    )
    assert any("metadata fields" in msg for msg in error_messages), (
        "Expected error about invalid metadata fields"
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_excessive_json_depth():
    """Tests error for excessive nesting depth in JSON patch document."""
    from arize.pandas.tracing.validation.metadata.value_validation import (
        MAX_JSON_NESTING_DEPTH,
    )

    # Create a deeply nested structure that exceeds the max depth
    nested_value = {}
    current = nested_value
    for _ in range(MAX_JSON_NESTING_DEPTH + 2):
        current["level"] = {}
        current = current["level"]

    # Create a dataframe with the excessively nested structure
    metadata_df = get_valid_metadata_df(1)
    metadata_df["patch_document"] = [nested_value]

    errors = validate_values(metadata_dataframe=metadata_df)
    assert len(errors) > 0, "Expected error for excessive JSON depth"
    assert any(
        "depth" in str(error).lower() or "nest" in str(error).lower()
        for error in errors
    ), "Expected error message about excessive nesting depth"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_excessive_json_size():
    """Tests error for excessive JSON string size in patch document."""
    from unittest.mock import patch

    from arize.pandas.tracing.validation.metadata.value_validation import (
        validate_values as original_validate,
    )

    # Create a large string that's just under the limit for testing purposes
    # (making it too large would be impractical for a unit test)
    test_size = 100000  # Use a smaller but still substantial size for the test
    large_string = "x" * test_size

    # Create a dataframe with a large JSON object
    metadata_df = get_valid_metadata_df(1)
    metadata_df["patch_document"] = [{"large_field": large_string}]

    # This should pass since it's under the max size
    errors_valid = validate_values(metadata_dataframe=metadata_df)
    assert len(errors_valid) == 0, (
        "Expected no errors for JSON under size limit"
    )

    # Use a direct patching approach instead of module reloading
    # Create a test with a much smaller size limit
    small_size_limit = (
        1000  # A very small limit that will definitely trigger the validation
    )

    # Create a string that's definitely bigger than our small test limit
    oversized_string = "x" * (small_size_limit + 500)

    test_df = get_valid_metadata_df(1)
    test_df["patch_document"] = [{"oversized_field": oversized_string}]

    # Patch the MAX_JSON_STRING_SIZE constant inside the validate_values function
    def patched_validate(
        metadata_dataframe, patch_document_column_name="patch_document"
    ):
        with patch(
            "arize.pandas.tracing.validation.metadata.value_validation.MAX_JSON_STRING_SIZE",
            small_size_limit,
        ):
            # Call the original function with our patched constant
            return original_validate(
                metadata_dataframe, patch_document_column_name
            )

    # Run with our patched function
    errors_with_small_limit = patched_validate(test_df)

    # Now we should get a size validation error
    assert len(errors_with_small_limit) > 0, (
        "Expected error for oversized JSON with small limit"
    )
    # Look for errors mentioning size
    size_errors = [
        e
        for e in errors_with_small_limit
        if "size" in str(e).lower() or "oversized" in str(e).lower()
    ]
    assert len(size_errors) > 0, (
        "Expected error message about size or oversized patch"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
