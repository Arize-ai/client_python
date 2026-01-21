import sys
import uuid

import pandas as pd
import pytest

if sys.version_info >= (3, 8):
    from arize.pandas.tracing.columns import SPAN_SPAN_ID_COL
    from arize.pandas.tracing.validation.metadata.argument_validation import (
        validate_argument_types,
    )
    from arize.pandas.tracing.validation.metadata.dataframe_form_validation import (
        validate_dataframe_form,
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


# Tests for argument_validation.py
@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_valid_argument_types():
    """Tests that valid arguments pass validation."""
    metadata_df = get_valid_metadata_df()
    project_name = "valid-project"

    errors = validate_argument_types(
        metadata_dataframe=metadata_df, project_name=project_name
    )
    assert len(errors) == 0, "Expected no validation errors for valid arguments"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_dataframe_type():
    """Tests error for non-DataFrame metadata_dataframe."""
    # Test with a dictionary instead of a DataFrame
    invalid_df = {"column": "value"}
    project_name = "valid-project"

    errors = validate_argument_types(
        metadata_dataframe=invalid_df, project_name=project_name
    )
    assert len(errors) > 0, (
        "Expected error for non-DataFrame metadata_dataframe"
    )
    assert any("dataframe" in str(error).lower() for error in errors)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_invalid_project_name_type():
    """Tests error for non-string project_name."""
    metadata_df = get_valid_metadata_df()

    # Test with None project_name
    errors_none = validate_argument_types(
        metadata_dataframe=metadata_df, project_name=None
    )
    assert len(errors_none) > 0, "Expected error for None project_name"
    assert any("project_name" in str(error).lower() for error in errors_none)

    # Test with empty string project_name
    errors_empty = validate_argument_types(
        metadata_dataframe=metadata_df, project_name=""
    )
    assert len(errors_empty) > 0, "Expected error for empty project_name"
    assert any("project_name" in str(error).lower() for error in errors_empty)

    # Test with non-string project_name
    errors_nonstring = validate_argument_types(
        metadata_dataframe=metadata_df, project_name=123
    )
    assert len(errors_nonstring) > 0, (
        "Expected error for non-string project_name"
    )
    assert any(
        "project_name" in str(error).lower() for error in errors_nonstring
    )


# Tests for dataframe_form_validation.py
@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_valid_dataframe_form():
    """Tests that a DataFrame with correct form passes validation."""
    metadata_df = get_valid_metadata_df()

    errors = validate_dataframe_form(metadata_dataframe=metadata_df)
    assert len(errors) == 0, (
        "Expected no validation errors for valid DataFrame form"
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_empty_dataframe():
    """Tests error for empty DataFrame."""
    empty_df = pd.DataFrame(columns=[SPAN_SPAN_ID_COL.name, "patch_document"])

    errors = validate_dataframe_form(metadata_dataframe=empty_df)
    assert len(errors) > 0, "Expected error for empty DataFrame"
    assert any("empty" in str(error).lower() for error in errors)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_missing_span_id_column():
    """Tests error for missing span_id column."""
    metadata_df = get_valid_metadata_df()
    df_no_span_id = metadata_df.drop(columns=[SPAN_SPAN_ID_COL.name])

    errors = validate_dataframe_form(metadata_dataframe=df_no_span_id)
    assert len(errors) > 0, "Expected error for missing span_id column"
    assert any(SPAN_SPAN_ID_COL.name in str(error) for error in errors)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_missing_patch_document_column():
    """Tests error for missing patch_document column."""
    metadata_df = get_valid_metadata_df()
    df_no_patch = metadata_df.drop(columns=["patch_document"])

    errors = validate_dataframe_form(metadata_dataframe=df_no_patch)
    assert len(errors) > 0, "Expected error for missing patch_document column"
    assert any("patch_document" in str(error) for error in errors)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_custom_patch_document_column():
    """Tests that custom patch_document_column_name works correctly."""
    metadata_df = get_valid_metadata_df()
    custom_name = "my_custom_patches"
    df_custom = metadata_df.rename(columns={"patch_document": custom_name})

    # Should fail with default column name
    errors_default = validate_dataframe_form(metadata_dataframe=df_custom)
    assert len(errors_default) > 0, "Expected error with default column name"

    # Should pass with custom column name
    errors_custom = validate_dataframe_form(
        metadata_dataframe=df_custom, patch_document_column_name=custom_name
    )
    assert len(errors_custom) == 0, "Expected no errors with custom column name"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_null_values():
    """Tests error for null values in required columns."""
    metadata_df = get_valid_metadata_df()

    # Test null span_id
    df_null_span_id = metadata_df.copy()
    df_null_span_id.loc[0, SPAN_SPAN_ID_COL.name] = None

    errors_span = validate_dataframe_form(metadata_dataframe=df_null_span_id)
    assert len(errors_span) > 0, "Expected error for null span_id"
    assert any("null values" in str(error).lower() for error in errors_span)

    # Test null patch_document
    df_null_patch = metadata_df.copy()
    df_null_patch.loc[0, "patch_document"] = None

    errors_patch = validate_dataframe_form(metadata_dataframe=df_null_patch)
    assert len(errors_patch) > 0, "Expected error for null patch_document"
    assert any("null values" in str(error).lower() for error in errors_patch)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_metadata_fields_only():
    """Tests that a DataFrame with only metadata fields passes validation."""
    span_id = str(uuid.uuid4())
    metadata_df = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: [span_id],
            "attributes.metadata.tag": ["important"],
            "attributes.metadata.priority": ["high"],
        }
    )

    errors = validate_dataframe_form(metadata_dataframe=metadata_df)
    assert len(errors) == 0, "Expected no errors with only metadata fields"

    # Test with null values in all metadata fields (should fail)
    df_null_fields = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: [span_id],
            "attributes.metadata.tag": [None],
            "attributes.metadata.priority": [None],
        }
    )

    errors_null = validate_dataframe_form(metadata_dataframe=df_null_fields)
    assert len(errors_null) > 0, (
        "Expected error when all metadata fields are null"
    )
    assert any("null values" in str(error).lower() for error in errors_null)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_combined_metadata_approach():
    """Tests validation with both metadata fields and patch_document."""
    span_id = str(uuid.uuid4())
    metadata_df = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: [span_id],
            "attributes.metadata.visible": [True],
            "attributes.metadata.tags": [["tag1", "tag2"]],
            "patch_document": ['{"nested": {"value": 42}}'],
        }
    )

    errors = validate_dataframe_form(metadata_dataframe=metadata_df)
    assert len(errors) == 0, "Expected no errors with combined approach"

    # Test with null patch_document but valid metadata fields (should pass)
    df_null_patch = metadata_df.copy()
    df_null_patch.loc[0, "patch_document"] = None

    errors_null_patch = validate_dataframe_form(
        metadata_dataframe=df_null_patch
    )
    assert len(errors_null_patch) > 0, (
        "Expected error with null patch document even with valid metadata fields"
    )

    # Test with no metadata fields, no patch document, and only span_id (should fail)
    df_no_metadata = pd.DataFrame({SPAN_SPAN_ID_COL.name: [span_id]})

    errors_no_metadata = validate_dataframe_form(
        metadata_dataframe=df_no_metadata
    )
    assert len(errors_no_metadata) > 0, (
        "Expected error with no metadata content"
    )
    assert any(
        "missing metadata columns" in str(error).lower()
        for error in errors_no_metadata
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
