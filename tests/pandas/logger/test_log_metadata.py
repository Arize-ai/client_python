import json
import sys
import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from arize.pandas.logger import Client
from arize.pandas.proto.requests_pb2 import WriteSpanAttributesMetadataResponse
from arize.utils.errors import AuthError

# Conditionally import tracing/validation components only for Python 3.8+
if sys.version_info >= (3, 8):
    from arize.pandas.tracing.columns import SPAN_SPAN_ID_COL
    from arize.pandas.validation.errors import (
        InvalidProjectName,
        InvalidTypeColumns,
        MissingColumns,
        ValidationFailure,
    )
else:
    # Define dummy classes/variables for Python < 3.8
    class ValidationFailure(Exception):
        pass

    class MissingColumns(Exception):
        pass

    class InvalidTypeColumns(Exception):
        pass

    class InvalidProjectName(Exception):
        pass

    # Dummy constants/objects needed for parsing even if tests are skipped
    SPAN_SPAN_ID_COL = type("SpanColumn", (), {"name": "context.span_id"})


# Sample test data
SAMPLE_SPAN_ID = str(uuid.uuid4())
SAMPLE_SPAN_ID_2 = str(uuid.uuid4())
SIMPLE_PATCH = {"key1": "value1", "nested": {"bool_value": True}}
SIMPLE_PATCH_2 = {"key2": "value2", "tags": ["tag1", "tag2"]}


class MockFlightSession:
    """Mocks the pyarrow.flight.FlightClient interaction for testing."""

    def __init__(self, *args, **kwargs):
        self.call_options = None

    def connect(self):
        mock_client = MagicMock()
        mock_client.do_put.return_value = (MagicMock(), MagicMock())
        mock_response = WriteSpanAttributesMetadataResponse()
        mock_response.spans_processed = 2
        mock_response.spans_updated = 5
        mock_response.spans_failed = 0

        # Create a sample error to test error handling
        mock_error = mock_response.SpanError()
        mock_error.span_id = "test-span-id"
        mock_error.error_message = "Test error message"
        mock_response.errors.append(mock_error)

        mock_client.do_put.return_value[1].read.return_value = MagicMock(
            to_pybytes=lambda: mock_response.SerializeToString()
        )
        return mock_client

    def close(self):
        pass


def get_metadata_df(num_rows=2, invalid_data=False):
    """Generates a sample pandas DataFrame for metadata update testing."""
    data = {
        SPAN_SPAN_ID_COL.name: [SAMPLE_SPAN_ID, SAMPLE_SPAN_ID_2][:num_rows],
        "patch_document": [
            json.dumps(SIMPLE_PATCH),
            json.dumps(SIMPLE_PATCH_2),
        ][:num_rows],
        "extra_column_to_be_ignored": [1, 2][:num_rows],
    }

    if invalid_data:
        # Create some invalid data for testing validation
        if num_rows > 0:
            data[SPAN_SPAN_ID_COL.name][0] = ""  # Empty span ID
        if num_rows > 1:
            data["patch_document"][1] = "{invalid-json"  # Invalid JSON

    return pd.DataFrame(data)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_metadata_success():
    """Tests successful update of span metadata with valid inputs."""
    client = Client(developer_key="dev_key", space_id="space_id")
    test_df = get_metadata_df()
    project_name = "test_project"

    try:
        response = client.log_metadata(
            dataframe=test_df,
            project_name=project_name,
            validate=True,
            verbose=False,
        )
        assert isinstance(response, dict)
        # The mock returns 5 for spans_updated in the prototype
        assert response.get("spans_updated") == 5
        # Check that spans_processed is correctly populated from the mock
        assert response.get("spans_processed") == 2
        # Check that spans_failed is correctly populated from the mock
        assert response.get("spans_failed") == 0
        # Verify errors array structure
        assert "errors" in response
        assert len(response["errors"]) == 1
        assert response["errors"][0]["span_id"] == "test-span-id"
        assert response["errors"][0]["error_message"] == "Test error message"
    except Exception as e:
        pytest.fail(f"log_metadata raised an unexpected exception: {e}")


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_log_metadata_missing_auth():
    """Tests that AuthError is raised if developer_key or space_id is missing."""
    test_df = get_metadata_df()
    project_name = "test_project"

    client_no_dev_key = Client(space_id="space_id")
    with pytest.raises(AuthError) as excinfo:
        client_no_dev_key.log_metadata(
            dataframe=test_df, project_name=project_name
        )
    assert excinfo.value.missing_developer_key is True
    assert excinfo.value.missing_space_id is False
    assert excinfo.value.method_name == "log_metadata"

    client_no_space_id = Client(developer_key="dev_key")
    with pytest.raises(AuthError) as excinfo:
        client_no_space_id.log_metadata(
            dataframe=test_df, project_name=project_name
        )
    assert excinfo.value.missing_space_id is True
    assert excinfo.value.missing_developer_key is False
    assert excinfo.value.method_name == "log_metadata"

    client_no_auth = Client()
    with pytest.raises(AuthError) as excinfo:
        client_no_auth.log_metadata(
            dataframe=test_df, project_name=project_name
        )
    assert excinfo.value.missing_space_id is True
    assert excinfo.value.missing_developer_key is True
    assert excinfo.value.method_name == "log_metadata"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_metadata_missing_project_name():
    """Tests that ValidationFailure is raised if project_name is missing."""
    client = Client(developer_key="dev_key", space_id="space_id")
    test_df = get_metadata_df()
    with pytest.raises(ValidationFailure) as excinfo:
        client.log_metadata(dataframe=test_df, project_name=None)
    error_messages = [str(e) for e in excinfo.value.errors]
    assert any(
        "project_name must be a non-empty string" in str(msg).lower()
        for msg in error_messages
    )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_metadata_missing_span_id():
    """Tests that ValidationFailure is raised if context.span_id is missing."""
    client = Client(developer_key="dev_key", space_id="space_id")
    test_df = get_metadata_df()
    test_df_no_span_id = test_df.drop(columns=[SPAN_SPAN_ID_COL.name])
    project_name = "test_project_missing_span_id"
    with pytest.raises(ValidationFailure) as excinfo:
        client.log_metadata(
            dataframe=test_df_no_span_id, project_name=project_name
        )
    assert SPAN_SPAN_ID_COL.name in str(excinfo.value.errors[0])


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_metadata_missing_patch_document():
    """Tests that ValidationFailure is raised if patch_document is missing."""
    client = Client(developer_key="dev_key", space_id="space_id")
    test_df = get_metadata_df()
    test_df_no_patch = test_df.drop(columns=["patch_document"])
    project_name = "test_project_missing_patch"

    # Add at least one metadata attribute column to test that branch
    test_df_no_patch["attributes.metadata.test_field"] = [
        "test_value",
        "test_value2",
    ]

    try:
        response = client.log_metadata(
            dataframe=test_df_no_patch, project_name=project_name
        )
        # Should succeed because we added an attributes.metadata.* column
        assert isinstance(response, dict)
        assert "spans_updated" in response
        assert "spans_processed" in response
        assert "spans_failed" in response
        assert "errors" in response
    except Exception as e:
        pytest.fail(
            f"log_metadata with metadata attribute columns failed unexpectedly: {e}"
        )

    # Now test without any metadata columns - should raise ValueError
    test_df_no_metadata = test_df.drop(columns=["patch_document"])
    with pytest.raises(ValueError) as excinfo:
        client.log_metadata(
            dataframe=test_df_no_metadata, project_name=project_name
        )
    assert "No metadata fields found" in str(excinfo.value)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_metadata_custom_column_name():
    """Tests that custom patch_document_column_name works correctly."""
    client = Client(developer_key="dev_key", space_id="space_id")
    test_df = get_metadata_df()
    custom_name = "my_custom_patches"
    test_df = test_df.rename(columns={"patch_document": custom_name})
    project_name = "test_project_custom_column"

    try:
        response = client.log_metadata(
            dataframe=test_df,
            project_name=project_name,
            patch_document_column_name=custom_name,
            validate=True,
        )
        assert isinstance(response, dict)
        # The mock returns 5 for spans_updated in the prototype
        assert response.get("spans_updated") == 5
        # Check that spans_processed is correctly populated from the mock
        assert response.get("spans_processed") == 2
        # Check that spans_failed is correctly populated from the mock
        assert response.get("spans_failed") == 0
        # Verify errors array structure
        assert "errors" in response
        assert len(response["errors"]) == 1
        assert response["errors"][0]["span_id"] == "test-span-id"
        assert response["errors"][0]["error_message"] == "Test error message"
    except Exception as e:
        pytest.fail(f"log_metadata with custom column raised an exception: {e}")


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_metadata_invalid_data():
    """Tests that ValidationFailure is raised for invalid data."""
    client = Client(developer_key="dev_key", space_id="space_id")

    # Test with empty span ID
    test_df_empty_span = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: [""],  # Empty span ID
            "patch_document": ['{"valid": "json"}'],
        }
    )

    project_name = "test_project_invalid_data"

    with pytest.raises(ValidationFailure) as excinfo:
        client.log_metadata(
            dataframe=test_df_empty_span, project_name=project_name
        )

    error_messages = [str(e) for e in excinfo.value.errors]
    assert any(
        "span id" in msg.lower() for msg in error_messages
    ), "Expected error about invalid span ID"

    # Test with invalid JSON
    test_df_invalid_json = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: [SAMPLE_SPAN_ID],
            "patch_document": ["{invalid-json"],
        }
    )

    with pytest.raises(ValidationFailure) as excinfo:
        client.log_metadata(
            dataframe=test_df_invalid_json, project_name=project_name
        )

    error_messages = [str(e) for e in excinfo.value.errors]
    # The error message now includes "invalid json" or similar text
    assert any(
        "invalid json" in msg.lower() or "json" in msg.lower()
        for msg in error_messages
    ), "Expected error about invalid JSON"


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.Client._log_arrow_flight")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_metadata_dict_conversion(mock_log_arrow_flight):
    """Tests that dictionary patch documents are properly converted to JSON strings."""
    # Create a properly structured response with the updated fields
    mock_response = WriteSpanAttributesMetadataResponse()
    mock_response.spans_processed = 2
    mock_response.spans_updated = 2
    mock_response.spans_failed = 0

    # Create a sample error
    mock_error = mock_response.SpanError()
    mock_error.span_id = "test-span-id"
    mock_error.error_message = "Test error message"
    mock_response.errors.append(mock_error)

    mock_log_arrow_flight.return_value = mock_response

    client = Client(developer_key="dev_key", space_id="space_id")

    # Create a DataFrame with dict objects instead of JSON strings
    test_df = pd.DataFrame(
        {
            SPAN_SPAN_ID_COL.name: [SAMPLE_SPAN_ID, SAMPLE_SPAN_ID_2],
            "patch_document": [
                SIMPLE_PATCH,
                SIMPLE_PATCH_2,
            ],  # Dict objects, not strings
        }
    )

    project_name = "test_project_dict_conversion"

    try:
        response = client.log_metadata(
            dataframe=test_df, project_name=project_name
        )
        assert isinstance(response, dict)
        assert response.get("spans_updated") == 2
        assert response.get("spans_processed") == 2
        assert "errors" in response
        # Check that we have the expected error structure
        assert len(response["errors"]) == 1
        assert "span_id" in response["errors"][0]
        assert "error_message" in response["errors"][0]
    except Exception as e:
        pytest.fail(f"log_metadata with dict objects raised an exception: {e}")

    # Verify the conversion happened correctly
    mock_log_arrow_flight.assert_called_once()
    call_args, call_kwargs = mock_log_arrow_flight.call_args
    pa_table = call_kwargs.get("pa_table", call_args[0] if call_args else None)
    assert pa_table is not None
    assert isinstance(pa_table, pa.Table)

    # Convert back to pandas to check
    result_df = pa_table.to_pandas()

    # Check patch documents were converted to strings
    for idx, patch_doc in enumerate(result_df["patch_document"]):
        assert isinstance(patch_doc, str)
        # Verify we can parse it back to JSON
        parsed = json.loads(patch_doc)
        if idx == 0:
            assert parsed == SIMPLE_PATCH
        else:
            assert parsed == SIMPLE_PATCH_2


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
@patch("arize.pandas.logger.Client._log_arrow_flight")
@patch("arize.pandas.logger.FlightSession", new=MockFlightSession)
def test_log_metadata_empty_dataframe(mock_log_arrow_flight):
    """Tests that logging an empty DataFrame raises ValidationFailure."""
    # We won't reach the arrow_flight call for empty dataframes
    mock_log_arrow_flight.return_value = WriteSpanAttributesMetadataResponse()

    client = Client(developer_key="dev_key", space_id="space_id")
    empty_df = pd.DataFrame(
        columns=[
            SPAN_SPAN_ID_COL.name,
            "patch_document",
        ]
    )
    project_name = "test_project_empty_df"

    with pytest.raises(ValidationFailure):
        client.log_metadata(dataframe=empty_df, project_name=project_name)

    mock_log_arrow_flight.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
