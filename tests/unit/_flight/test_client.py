"""Unit tests for arize._flight.client module.

This file contains all tests for ArizeFlightClient:
- Low-level internals (initialization, connection, properties, passthrough methods)
- High-level workflows (log_arrow_table, create_dataset, get_dataset_examples, etc.)

All tests use mocks and are marked with @pytest.mark.unit.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pyarrow as pa
import pytest
from pyarrow import flight

from arize._flight.client import (
    ArizeFlightClient,
    _get_pb_flight_doput_request,
    append_to_pyarrow_metadata,
)
from arize._flight.types import FlightRequestType
from arize._generated.protocol.flight import flight_pb2

if TYPE_CHECKING:
    from collections.abc import Iterator


# ==================== Helper Functions ====================


def create_context_mock_writer() -> MagicMock:
    """Create a mock writer that supports context manager protocol.

    It uses MagicMock since it automatically supports context managers,
    we just need to set return value
    """
    mock_writer = MagicMock()
    mock_writer.__enter__.return_value = mock_writer
    return mock_writer


# ==================== Low-Level Tests ====================


@pytest.mark.unit
class TestArizeFlightClientInit:
    """Test ArizeFlightClient initialization."""

    def test_init_with_all_params(self) -> None:
        """Test client initialization with all parameters."""
        client = ArizeFlightClient(
            api_key="test_key",
            host="example.com",
            port=443,
            scheme="https",
            max_chunksize=2000,
            request_verify=False,
        )
        assert client.api_key == "test_key"
        assert client.host == "example.com"
        assert client.port == 443
        assert client.scheme == "https"
        assert client.max_chunksize == 2000
        assert client.request_verify is False

    def test_init_internal_client_is_none(
        self, flight_client: ArizeFlightClient
    ) -> None:
        """Test that internal _client is None on initialization."""
        assert object.__getattribute__(flight_client, "_client") is None

    def test_frozen_dataclass(self, flight_client: ArizeFlightClient) -> None:
        """Test that ArizeFlightClient is frozen."""
        with pytest.raises(AttributeError):
            flight_client.api_key = "new_key"  # type: ignore[misc]


@pytest.mark.unit
class TestArizeFlightClientProperties:
    """Test ArizeFlightClient properties."""

    def test_headers_property(self, flight_client: ArizeFlightClient) -> None:
        """Test that headers property returns correct headers."""
        headers = flight_client.headers
        assert isinstance(headers, list)
        assert len(headers) == 5
        assert (b"origin", b"arize-logging-client") in headers
        assert (b"auth-token-bin", b"test_api_key") in headers
        assert (b"sdk-language", b"python") in headers

    def test_headers_contain_version_info(
        self, flight_client: ArizeFlightClient
    ) -> None:
        """Test that headers contain version information."""
        headers = flight_client.headers
        headers_dict = dict(headers)
        assert b"language-version" in headers_dict
        assert b"sdk-version" in headers_dict

    def test_call_options_property(
        self, flight_client: ArizeFlightClient
    ) -> None:
        """Test that call_options property returns FlightCallOptions."""
        call_options = flight_client.call_options
        assert isinstance(call_options, flight.FlightCallOptions)


@pytest.mark.unit
class TestArizeFlightClientConnection:
    """Test ArizeFlightClient connection management."""

    # Using `@patch` decorator instead of `with patch()` context manager because:
    # 1. Mock applies to entire test function (cleaner for whole-function mocking)
    # 2. Mock is passed as a parameter (no need for setup before patching)
    # 3. Standard pattern for non-frozen dataclasses
    # Alternative: Use `with patch()` when mock only applies to part of test,
    # or when working with frozen dataclasses (see _exporter tests)
    @patch("arize._flight.client.flight.FlightClient")
    def test_ensure_client_creates_client(
        self,
        mock_flight_client_class: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test that _ensure_client creates a new FlightClient."""
        mock_client_instance = Mock()
        mock_flight_client_class.return_value = mock_client_instance

        result = flight_client._ensure_client()

        assert result == mock_client_instance
        mock_flight_client_class.assert_called_once_with(
            location="https://test-host.com:443",
            disable_server_verification=False,
        )

    @patch("arize._flight.client.flight.FlightClient")
    def test_ensure_client_returns_cached_client(
        self,
        mock_flight_client_class: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test that _ensure_client returns cached client on subsequent calls."""
        mock_client_instance = Mock()
        mock_flight_client_class.return_value = mock_client_instance

        first_call = flight_client._ensure_client()
        second_call = flight_client._ensure_client()

        assert first_call == second_call
        assert mock_flight_client_class.call_count == 1

    @patch("arize._flight.client.flight.FlightClient")
    def test_ensure_client_disables_cert_for_localhost(
        self,
        mock_flight_client_class: Mock,
        flight_client_localhost: ArizeFlightClient,
    ) -> None:
        """Test that TLS verification is disabled for localhost."""
        mock_client_instance = Mock()
        mock_flight_client_class.return_value = mock_client_instance

        flight_client_localhost._ensure_client()

        mock_flight_client_class.assert_called_once_with(
            location="http://localhost:8080",
            disable_server_verification=True,
        )

    @patch("arize._flight.client.flight.FlightClient")
    def test_ensure_client_disables_cert_when_verify_false(
        self,
        mock_flight_client_class: Mock,
    ) -> None:
        """Test that TLS verification is disabled when request_verify=False."""
        client = ArizeFlightClient(
            api_key="test_key",
            host="example.com",
            port=443,
            scheme="https",
            max_chunksize=1000,
            request_verify=False,
        )
        mock_client_instance = Mock()
        mock_flight_client_class.return_value = mock_client_instance

        client._ensure_client()

        mock_flight_client_class.assert_called_once_with(
            location="https://example.com:443",
            disable_server_verification=True,
        )

    @patch("arize._flight.client.flight.FlightClient")
    def test_close_closes_client(
        self,
        mock_flight_client_class: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test that close method closes the underlying client."""
        mock_client_instance = Mock()
        mock_flight_client_class.return_value = mock_client_instance

        flight_client._ensure_client()
        flight_client.close()

        mock_client_instance.close.assert_called_once()
        assert object.__getattribute__(flight_client, "_client") is None

    def test_close_when_no_client(
        self, flight_client: ArizeFlightClient
    ) -> None:
        """Test that close method does nothing when no client exists."""
        flight_client.close()  # Should not raise an exception


@pytest.mark.unit
class TestArizeFlightClientContextManager:
    """Test ArizeFlightClient context manager."""

    @patch("arize._flight.client.flight.FlightClient")
    def test_context_manager_enter(
        self,
        mock_flight_client_class: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test context manager __enter__ initializes client."""
        mock_client_instance = Mock()
        mock_flight_client_class.return_value = mock_client_instance

        with flight_client as client:
            assert client == flight_client
            assert object.__getattribute__(client, "_client") is not None

    @patch("arize._flight.client.flight.FlightClient")
    def test_context_manager_exit(
        self,
        mock_flight_client_class: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test context manager __exit__ closes client."""
        mock_client_instance = Mock()
        mock_flight_client_class.return_value = mock_client_instance

        with flight_client:
            pass

        mock_client_instance.close.assert_called_once()
        assert object.__getattribute__(flight_client, "_client") is None

    @patch("arize._flight.client.flight.FlightClient")
    @patch("arize._flight.client.logger")
    def test_context_manager_exit_with_exception(
        self,
        mock_logger: Mock,
        mock_flight_client_class: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test context manager __exit__ logs exception and closes client."""
        mock_client_instance = Mock()
        mock_flight_client_class.return_value = mock_client_instance

        with pytest.raises(ValueError), flight_client:
            raise ValueError("Test exception")

        mock_logger.error.assert_called_once()
        mock_client_instance.close.assert_called_once()


@pytest.mark.unit
class TestArizeFlightClientPassthroughMethods:
    """Test ArizeFlightClient passthrough methods."""

    @patch("arize._flight.client.flight.FlightClient")
    def test_get_flight_info(
        self,
        mock_flight_client_class: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test get_flight_info passthrough method."""
        mock_client_instance = Mock()
        mock_flight_client_class.return_value = mock_client_instance
        mock_flight_info = Mock()
        mock_client_instance.get_flight_info.return_value = mock_flight_info

        result = flight_client.get_flight_info("test_arg")

        assert result == mock_flight_info
        mock_client_instance.get_flight_info.assert_called_once()

    @patch("arize._flight.client.flight.FlightClient")
    def test_do_get(
        self,
        mock_flight_client_class: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test do_get passthrough method."""
        mock_client_instance = Mock()
        mock_flight_client_class.return_value = mock_client_instance
        mock_stream_reader = Mock()
        mock_client_instance.do_get.return_value = mock_stream_reader

        result = flight_client.do_get("test_arg")

        assert result == mock_stream_reader
        mock_client_instance.do_get.assert_called_once()

    @patch("arize._flight.client.flight.FlightClient")
    def test_do_put(
        self,
        mock_flight_client_class: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test do_put passthrough method."""
        mock_client_instance = Mock()
        mock_flight_client_class.return_value = mock_client_instance
        mock_writer = Mock()
        mock_reader = Mock()
        mock_client_instance.do_put.return_value = (mock_writer, mock_reader)

        result = flight_client.do_put("test_arg")

        assert result == (mock_writer, mock_reader)
        mock_client_instance.do_put.assert_called_once()

    @patch("arize._flight.client.flight.FlightClient")
    def test_do_action(
        self,
        mock_flight_client_class: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test do_action passthrough method."""
        mock_client_instance = Mock()
        mock_flight_client_class.return_value = mock_client_instance
        mock_results: Iterator[flight.Result] = iter([Mock()])
        mock_client_instance.do_action.return_value = mock_results

        result = flight_client.do_action("test_arg")

        assert result == mock_results
        mock_client_instance.do_action.assert_called_once()


@pytest.mark.unit
class TestAppendToPyarrowMetadata:
    """Test append_to_pyarrow_metadata function."""

    def test_append_to_empty_metadata(self) -> None:
        """Test appending metadata to schema with no existing metadata."""
        schema = pa.schema([("col1", pa.int32())])
        new_metadata = {"key1": b"value1", "key2": b"value2"}

        result = append_to_pyarrow_metadata(schema, new_metadata)

        assert result.metadata[b"key1"] == b"value1"
        assert result.metadata[b"key2"] == b"value2"

    def test_append_to_existing_metadata(self) -> None:
        """Test appending metadata to schema with existing metadata."""
        schema = pa.schema(
            [("col1", pa.int32())], metadata={"existing": b"value"}
        )
        new_metadata = {"key1": b"value1"}

        result = append_to_pyarrow_metadata(schema, new_metadata)

        assert result.metadata[b"existing"] == b"value"
        assert result.metadata[b"key1"] == b"value1"

    def test_conflicting_keys_raises_error(self) -> None:
        """Test that conflicting keys raise KeyError."""
        schema = pa.schema(
            [("col1", pa.int32())], metadata={b"key1": b"original"}
        )
        new_metadata = {b"key1": b"new_value"}

        with pytest.raises(KeyError) as exc_info:
            append_to_pyarrow_metadata(schema, new_metadata)

        assert "conflicting keys" in str(exc_info.value)

    def test_multiple_conflicting_keys(self) -> None:
        """Test error message includes all conflicting keys."""
        schema = pa.schema(
            [("col1", pa.int32())],
            metadata={b"key1": b"val1", b"key2": b"val2"},
        )
        new_metadata = {b"key1": b"new1", b"key2": b"new2"}

        with pytest.raises(KeyError) as exc_info:
            append_to_pyarrow_metadata(schema, new_metadata)

        error_msg = str(exc_info.value)
        assert "key1" in error_msg or "key2" in error_msg


@pytest.mark.unit
class TestGetPbFlightDoputRequest:
    """Test _get_pb_flight_doput_request function."""

    def test_evaluation_request_type(self) -> None:
        """Test creating evaluation request."""
        result = _get_pb_flight_doput_request(
            space_id="space123",
            request_type=FlightRequestType.EVALUATION,
            model_id="model123",
        )
        assert result.HasField("write_span_evaluation_request")
        assert result.write_span_evaluation_request.space_id == "space123"
        assert (
            result.write_span_evaluation_request.external_model_id == "model123"
        )

    def test_annotation_request_type(self) -> None:
        """Test creating annotation request."""
        result = _get_pb_flight_doput_request(
            space_id="space123",
            request_type=FlightRequestType.ANNOTATION,
            model_id="model123",
        )
        assert result.HasField("write_span_annotation_request")
        assert result.write_span_annotation_request.space_id == "space123"
        assert (
            result.write_span_annotation_request.external_model_id == "model123"
        )

    def test_metadata_request_type(self) -> None:
        """Test creating metadata request."""
        result = _get_pb_flight_doput_request(
            space_id="space123",
            request_type=FlightRequestType.METADATA,
            model_id="model123",
        )
        assert result.HasField("write_span_attributes_metadata_request")
        assert (
            result.write_span_attributes_metadata_request.space_id == "space123"
        )
        assert (
            result.write_span_attributes_metadata_request.external_model_id
            == "model123"
        )

    def test_log_experiment_data_request_type(self) -> None:
        """Test creating log experiment data request."""
        result = _get_pb_flight_doput_request(
            space_id="space123",
            request_type=FlightRequestType.LOG_EXPERIMENT_DATA,
            dataset_id="dataset123",
            experiment_name="exp1",
        )
        assert result.HasField("post_experiment_data")
        assert result.post_experiment_data.space_id == "space123"
        assert result.post_experiment_data.dataset_id == "dataset123"
        assert result.post_experiment_data.experiment_name == "exp1"

    def test_evaluation_without_model_id_raises_error(self) -> None:
        """Test that evaluation request without model_id raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _get_pb_flight_doput_request(
                space_id="space123",
                request_type=FlightRequestType.EVALUATION,
            )
        assert "Unsupported" in str(exc_info.value)

    def test_log_experiment_without_dataset_id_raises_error(self) -> None:
        """Test that experiment request without dataset_id raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _get_pb_flight_doput_request(
                space_id="space123",
                request_type=FlightRequestType.LOG_EXPERIMENT_DATA,
                experiment_name="exp1",
            )
        assert "Unsupported" in str(exc_info.value)

    def test_log_experiment_without_experiment_name_raises_error(
        self,
    ) -> None:
        """Test that experiment request without experiment_name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _get_pb_flight_doput_request(
                space_id="space123",
                request_type=FlightRequestType.LOG_EXPERIMENT_DATA,
                dataset_id="dataset123",
            )
        assert "Unsupported" in str(exc_info.value)


# ==================== High-Level Workflow Tests ====================


@pytest.mark.unit
class TestLogArrowTable:
    """Test log_arrow_table method workflows."""

    @pytest.mark.parametrize(
        "request_type,response_class,response_field,expected_value,extra_kwargs,needs_schema_mock",
        [
            (
                FlightRequestType.EVALUATION,
                flight_pb2.WriteSpanEvaluationResponse,
                "records_updated",
                3,
                {"project_name": "test_project"},
                True,
            ),
            (
                FlightRequestType.ANNOTATION,
                flight_pb2.WriteSpanAnnotationResponse,
                "records_updated",
                3,
                {"project_name": "test_project"},
                True,
            ),
            (
                FlightRequestType.METADATA,
                flight_pb2.WriteSpanAttributesMetadataResponse,
                "spans_updated",
                3,
                {"project_name": "test_project"},
                True,
            ),
            (
                FlightRequestType.LOG_EXPERIMENT_DATA,
                flight_pb2.PostExperimentDataResponse,
                "experiment_id",
                "exp_123",
                {"dataset_id": "dataset123", "experiment_name": "exp1"},
                False,
            ),
        ],
    )
    @patch("arize._flight.client.ArizeFlightClient.do_put")
    @patch("arize._flight.client.get_pb_schema_tracing")
    def test_log_arrow_table_success(
        self,
        mock_get_schema: Mock,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table: pa.Table,
        request_type: FlightRequestType,
        response_class: type,
        response_field: str,
        expected_value: str | int,
        extra_kwargs: dict,
        needs_schema_mock: bool,
    ) -> None:
        """Test successful logging for various request types."""
        # Setup schema mock for tracing requests
        if needs_schema_mock:
            mock_schema = Mock()
            mock_schema.SerializeToString.return_value = b"schema_bytes"
            mock_get_schema.return_value = mock_schema

        # Setup writer and response mocks
        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_response = Mock()

        # Create appropriate response protobuf
        response = response_class()
        setattr(response, response_field, expected_value)
        # Add dataset_id for experiment responses
        if request_type == FlightRequestType.LOG_EXPERIMENT_DATA:
            response.dataset_id = "dataset123"
        mock_response.to_pybytes.return_value = response.SerializeToString()
        mock_metadata_reader.read.return_value = mock_response

        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        # Execute
        result = flight_client.log_arrow_table(
            space_id="test_space",
            request_type=request_type,
            pa_table=sample_pa_table,
            **extra_kwargs,
        )

        # Verify
        assert isinstance(result, response_class)
        assert getattr(result, response_field) == expected_value
        mock_writer.write_table.assert_called_once()
        mock_writer.done_writing.assert_called_once()

        # Verify schema mock was called for tracing requests
        if needs_schema_mock:
            mock_get_schema.assert_called_once_with(project_name="test_project")

    @pytest.mark.parametrize(
        "request_type",
        [
            FlightRequestType.EVALUATION,
            FlightRequestType.ANNOTATION,
            FlightRequestType.METADATA,
        ],
    )
    def test_log_tracing_missing_project_name(
        self,
        flight_client: ArizeFlightClient,
        sample_pa_table: pa.Table,
        request_type: FlightRequestType,
    ) -> None:
        """Test that tracing requests without project_name raise ValueError."""
        with pytest.raises(ValueError, match="project_name is required"):
            flight_client.log_arrow_table(
                space_id="test_space",
                request_type=request_type,
                pa_table=sample_pa_table,
                project_name=None,
            )

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    @patch("arize._flight.client.get_pb_schema_tracing")
    def test_log_arrow_table_none_response(
        self,
        mock_get_schema: Mock,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table: pa.Table,
    ) -> None:
        """Test handling of None response from server."""
        # Setup mocks
        mock_schema = Mock()
        mock_schema.SerializeToString.return_value = b"schema_bytes"
        mock_get_schema.return_value = mock_schema

        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_metadata_reader.read.return_value = None

        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        # Execute
        result = flight_client.log_arrow_table(
            space_id="test_space",
            request_type=FlightRequestType.EVALUATION,
            pa_table=sample_pa_table,
            project_name="test_project",
        )

        # Verify
        assert result is None

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    @patch("arize._flight.client.get_pb_schema_tracing")
    def test_log_arrow_table_flight_exception(
        self,
        mock_get_schema: Mock,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table: pa.Table,
    ) -> None:
        """Test that Flight exceptions are wrapped in RuntimeError."""
        # Setup mocks
        mock_schema = Mock()
        mock_schema.SerializeToString.return_value = b"schema_bytes"
        mock_get_schema.return_value = mock_schema

        mock_do_put.side_effect = Exception("Flight connection failed")

        # Execute & Verify
        with pytest.raises(
            RuntimeError, match="Error logging arrow table to Arize"
        ):
            flight_client.log_arrow_table(
                space_id="test_space",
                request_type=FlightRequestType.EVALUATION,
                pa_table=sample_pa_table,
                project_name="test_project",
            )

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    @patch("arize._flight.client.get_pb_schema_tracing")
    def test_log_arrow_table_uses_max_chunksize(
        self,
        mock_get_schema: Mock,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table_large: pa.Table,
    ) -> None:
        """Test that max_chunksize is passed to write_table."""
        # Setup mocks
        mock_schema = Mock()
        mock_schema.SerializeToString.return_value = b"schema_bytes"
        mock_get_schema.return_value = mock_schema

        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_response = Mock()

        response = flight_pb2.WriteSpanEvaluationResponse()
        response.records_updated = 100
        mock_response.to_pybytes.return_value = response.SerializeToString()
        mock_metadata_reader.read.return_value = mock_response

        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        # Execute
        flight_client.log_arrow_table(
            space_id="test_space",
            request_type=FlightRequestType.EVALUATION,
            pa_table=sample_pa_table_large,
            project_name="test_project",
        )

        # Verify max_chunksize was used
        call_args = mock_writer.write_table.call_args
        assert call_args[0][1] == 1000  # max_chunksize

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    @patch("arize._flight.client.get_pb_schema_tracing")
    @patch("arize._flight.client.append_to_pyarrow_metadata")
    def test_log_tracing_request_appends_schema_metadata(
        self,
        mock_append_metadata: Mock,
        mock_get_schema: Mock,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table: pa.Table,
    ) -> None:
        """Test that tracing requests append schema metadata."""
        # Setup mocks
        mock_schema = Mock()
        mock_schema.SerializeToString.return_value = b"schema_bytes"
        mock_get_schema.return_value = mock_schema

        modified_schema = Mock()
        mock_append_metadata.return_value = modified_schema

        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_response = Mock()

        response = flight_pb2.WriteSpanEvaluationResponse()
        mock_response.to_pybytes.return_value = response.SerializeToString()
        mock_metadata_reader.read.return_value = mock_response

        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        # Execute
        flight_client.log_arrow_table(
            space_id="test_space",
            request_type=FlightRequestType.EVALUATION,
            pa_table=sample_pa_table,
            project_name="test_project",
        )

        # Verify metadata was appended
        mock_append_metadata.assert_called_once()
        call_args = mock_append_metadata.call_args[0]
        assert call_args[0] == sample_pa_table.schema


@pytest.mark.unit
class TestCreateDataset:
    """Test create_dataset method workflows."""

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_success(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table: pa.Table,
    ) -> None:
        """Test successful dataset creation."""
        # Setup mocks
        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_response = Mock()

        response = flight_pb2.CreateDatasetResponse()
        response.dataset_id = "dataset_12345"
        mock_response.to_pybytes.return_value = response.SerializeToString()
        mock_metadata_reader.read.return_value = mock_response

        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        # Execute
        result = flight_client.create_dataset(
            space_id="test_space",
            dataset_name="test_dataset",
            pa_table=sample_pa_table,
        )

        # Verify
        assert result == "dataset_12345"
        mock_writer.write_table.assert_called_once()
        mock_writer.done_writing.assert_called_once()

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_descriptor_format(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table: pa.Table,
    ) -> None:
        """Test that descriptor is correctly formatted with DoPutRequest."""
        # Setup mocks
        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_response = Mock()

        response = flight_pb2.CreateDatasetResponse()
        response.dataset_id = "dataset_123"
        mock_response.to_pybytes.return_value = response.SerializeToString()
        mock_metadata_reader.read.return_value = mock_response

        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        # Execute
        flight_client.create_dataset(
            space_id="test_space",
            dataset_name="test_dataset",
            pa_table=sample_pa_table,
        )

        # Verify descriptor was created correctly
        call_args = mock_do_put.call_args
        descriptor = call_args[0][0]

        # Decode and verify the descriptor contains CreateDatasetRequest
        descriptor_json = json.loads(descriptor.command.decode("utf-8"))
        assert "createDataset" in descriptor_json
        assert descriptor_json["createDataset"]["spaceId"] == "test_space"
        assert descriptor_json["createDataset"]["datasetName"] == "test_dataset"

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_none_response(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table: pa.Table,
    ) -> None:
        """Test handling of None response from server."""
        # Setup mocks
        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_metadata_reader.read.return_value = None

        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        # Execute
        result = flight_client.create_dataset(
            space_id="test_space",
            dataset_name="test_dataset",
            pa_table=sample_pa_table,
        )

        # Verify
        assert result is None

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_flight_exception(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table: pa.Table,
    ) -> None:
        """Test that Flight exceptions are wrapped in RuntimeError."""
        mock_do_put.side_effect = Exception("Flight connection failed")

        # Execute & Verify
        with pytest.raises(
            RuntimeError, match="Error logging arrow table to Arize"
        ):
            flight_client.create_dataset(
                space_id="test_space",
                dataset_name="test_dataset",
                pa_table=sample_pa_table,
            )

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_write_workflow(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table: pa.Table,
    ) -> None:
        """Test the complete write workflow: write_table → done_writing → read."""
        # Setup mocks
        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_response = Mock()

        response = flight_pb2.CreateDatasetResponse()
        response.dataset_id = "dataset_123"
        mock_response.to_pybytes.return_value = response.SerializeToString()
        mock_metadata_reader.read.return_value = mock_response

        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        # Execute
        flight_client.create_dataset(
            space_id="test_space",
            dataset_name="test_dataset",
            pa_table=sample_pa_table,
        )

        # Verify workflow order
        assert mock_writer.write_table.called
        assert mock_writer.done_writing.called
        assert mock_metadata_reader.read.called

        # Verify write_table was called before done_writing
        write_call_order = mock_writer.method_calls.index(
            ("write_table", (sample_pa_table, 1000), {})
        )
        done_call_order = mock_writer.method_calls.index(
            ("done_writing", (), {})
        )
        assert write_call_order < done_call_order

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_uses_max_chunksize(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table_large: pa.Table,
    ) -> None:
        """Test that max_chunksize is passed to write_table."""
        # Setup mocks
        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_response = Mock()

        response = flight_pb2.CreateDatasetResponse()
        response.dataset_id = "dataset_123"
        mock_response.to_pybytes.return_value = response.SerializeToString()
        mock_metadata_reader.read.return_value = mock_response

        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        # Execute
        flight_client.create_dataset(
            space_id="test_space",
            dataset_name="test_dataset",
            pa_table=sample_pa_table_large,
        )

        # Verify max_chunksize was used
        call_args = mock_writer.write_table.call_args
        assert call_args[0][1] == 1000


@pytest.mark.unit
class TestGetDatasetExamples:
    """Test get_dataset_examples method workflows."""

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_get_dataset_examples_success(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        sample_dataset_df: pd.DataFrame,
    ) -> None:
        """Test successful retrieval of dataset examples."""
        # Setup mocks
        mock_reader = Mock()
        mock_table = Mock()
        mock_table.to_pandas.return_value = sample_dataset_df
        mock_reader.read_all.return_value = mock_table
        mock_do_get.return_value = mock_reader

        # Execute
        result = flight_client.get_dataset_examples(
            space_id="test_space",
            dataset_id="dataset_123",
        )

        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_get_dataset_examples_with_version(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        sample_dataset_df: pd.DataFrame,
    ) -> None:
        """Test retrieval with specific version_id."""
        # Setup mocks
        mock_reader = Mock()
        mock_table = Mock()
        mock_table.to_pandas.return_value = sample_dataset_df
        mock_reader.read_all.return_value = mock_table
        mock_do_get.return_value = mock_reader

        # Execute
        result = flight_client.get_dataset_examples(
            space_id="test_space",
            dataset_id="dataset_123",
            dataset_version_id="v1",
        )

        # Verify
        assert isinstance(result, pd.DataFrame)

        # Verify version was included in request
        call_args = mock_do_get.call_args
        ticket = call_args[0][0]
        ticket_json = json.loads(ticket.ticket.decode("utf-8"))
        assert ticket_json["getDataset"]["datasetVersion"] == "v1"

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_get_dataset_examples_without_version(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        sample_dataset_df: pd.DataFrame,
    ) -> None:
        """Test retrieval without version_id (latest version)."""
        # Setup mocks
        mock_reader = Mock()
        mock_table = Mock()
        mock_table.to_pandas.return_value = sample_dataset_df
        mock_reader.read_all.return_value = mock_table
        mock_do_get.return_value = mock_reader

        # Execute
        result = flight_client.get_dataset_examples(
            space_id="test_space",
            dataset_id="dataset_123",
            dataset_version_id=None,
        )

        # Verify
        assert isinstance(result, pd.DataFrame)

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_get_dataset_examples_ticket_format(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        sample_dataset_df: pd.DataFrame,
    ) -> None:
        """Test that ticket is correctly formatted with DoGetRequest."""
        # Setup mocks
        mock_reader = Mock()
        mock_table = Mock()
        mock_table.to_pandas.return_value = sample_dataset_df
        mock_reader.read_all.return_value = mock_table
        mock_do_get.return_value = mock_reader

        # Execute
        flight_client.get_dataset_examples(
            space_id="test_space",
            dataset_id="dataset_123",
        )

        # Verify ticket format
        call_args = mock_do_get.call_args
        ticket = call_args[0][0]
        ticket_json = json.loads(ticket.ticket.decode("utf-8"))

        assert "getDataset" in ticket_json
        assert ticket_json["getDataset"]["spaceId"] == "test_space"
        assert ticket_json["getDataset"]["datasetId"] == "dataset_123"

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_get_dataset_examples_flight_exception(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test that Flight exceptions are wrapped in RuntimeError."""
        mock_do_get.side_effect = Exception("Flight connection failed")

        # Execute & Verify
        with pytest.raises(
            RuntimeError, match="Failed to get dataset id=dataset_123"
        ):
            flight_client.get_dataset_examples(
                space_id="test_space",
                dataset_id="dataset_123",
            )

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_get_dataset_examples_empty_dataset(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test handling of empty dataset."""
        # Setup mocks
        mock_reader = Mock()
        mock_table = Mock()
        empty_df = pd.DataFrame()
        mock_table.to_pandas.return_value = empty_df
        mock_reader.read_all.return_value = mock_table
        mock_do_get.return_value = mock_reader

        # Execute
        result = flight_client.get_dataset_examples(
            space_id="test_space",
            dataset_id="dataset_123",
        )

        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


@pytest.mark.unit
class TestGetExperimentRuns:
    """Test get_experiment_runs method workflows."""

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_get_experiment_runs_success(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        sample_experiment_df: pd.DataFrame,
    ) -> None:
        """Test successful retrieval of experiment runs."""
        # Setup mocks
        mock_reader = Mock()
        mock_table = Mock()
        mock_table.to_pandas.return_value = sample_experiment_df
        mock_reader.read_all.return_value = mock_table
        mock_do_get.return_value = mock_reader

        # Execute
        result = flight_client.get_experiment_runs(
            space_id="test_space",
            experiment_id="exp_123",
        )

        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_get_experiment_runs_ticket_format(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        sample_experiment_df: pd.DataFrame,
    ) -> None:
        """Test that ticket is correctly formatted with DoGetRequest."""
        # Setup mocks
        mock_reader = Mock()
        mock_table = Mock()
        mock_table.to_pandas.return_value = sample_experiment_df
        mock_reader.read_all.return_value = mock_table
        mock_do_get.return_value = mock_reader

        # Execute
        flight_client.get_experiment_runs(
            space_id="test_space",
            experiment_id="exp_123",
        )

        # Verify ticket format
        call_args = mock_do_get.call_args
        ticket = call_args[0][0]
        ticket_json = json.loads(ticket.ticket.decode("utf-8"))

        assert "getExperiment" in ticket_json
        assert ticket_json["getExperiment"]["spaceId"] == "test_space"
        assert ticket_json["getExperiment"]["experimentId"] == "exp_123"

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_get_experiment_runs_flight_exception(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test that Flight exceptions are wrapped in RuntimeError."""
        mock_do_get.side_effect = Exception("Flight connection failed")

        # Execute & Verify
        with pytest.raises(
            RuntimeError, match="Failed to get experiment id=exp_123"
        ):
            flight_client.get_experiment_runs(
                space_id="test_space",
                experiment_id="exp_123",
            )

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_get_experiment_runs_empty_results(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test handling of empty experiment results."""
        # Setup mocks
        mock_reader = Mock()
        mock_table = Mock()
        empty_df = pd.DataFrame()
        mock_table.to_pandas.return_value = empty_df
        mock_reader.read_all.return_value = mock_table
        mock_do_get.return_value = mock_reader

        # Execute
        result = flight_client.get_experiment_runs(
            space_id="test_space",
            experiment_id="exp_123",
        )

        # Verify
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_get_experiment_runs_read_workflow(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        sample_experiment_df: pd.DataFrame,
    ) -> None:
        """Test the complete read workflow: do_get → read_all → to_pandas."""
        # Setup mocks
        mock_reader = Mock()
        mock_table = Mock()
        mock_table.to_pandas.return_value = sample_experiment_df
        mock_reader.read_all.return_value = mock_table
        mock_do_get.return_value = mock_reader

        # Execute
        flight_client.get_experiment_runs(
            space_id="test_space",
            experiment_id="exp_123",
        )

        # Verify workflow
        mock_do_get.assert_called_once()
        mock_reader.read_all.assert_called_once()
        mock_table.to_pandas.assert_called_once()


@pytest.mark.unit
class TestInitExperiment:
    """Test init_experiment method workflows."""

    @patch("arize._flight.client.ArizeFlightClient.do_action")
    def test_init_experiment_success(
        self,
        mock_do_action: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test successful experiment initialization."""
        # Setup mocks
        mock_result = Mock()
        response = flight_pb2.CreateExperimentDBEntryResponse()
        response.experiment_id = "exp_12345"
        response.trace_model_name = "trace_model_67890"
        mock_result.body.to_pybytes.return_value = response.SerializeToString()

        mock_do_action.return_value = iter([mock_result])

        # Execute
        result = flight_client.init_experiment(
            space_id="test_space",
            dataset_id="dataset_123",
            experiment_name="test_experiment",
        )

        # Verify
        assert result is not None
        assert result[0] == "exp_12345"
        assert result[1] == "trace_model_67890"

    @patch("arize._flight.client.ArizeFlightClient.do_action")
    def test_init_experiment_action_format(
        self,
        mock_do_action: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test that action is correctly formatted with DoActionRequest."""
        # Setup mocks
        mock_result = Mock()
        response = flight_pb2.CreateExperimentDBEntryResponse()
        response.experiment_id = "exp_123"
        response.trace_model_name = "trace_model_456"
        mock_result.body.to_pybytes.return_value = response.SerializeToString()

        mock_do_action.return_value = iter([mock_result])

        # Execute
        flight_client.init_experiment(
            space_id="test_space",
            dataset_id="dataset_123",
            experiment_name="test_experiment",
        )

        # Verify action format
        call_args = mock_do_action.call_args
        action = call_args[0][0]

        assert action.type == "create_experiment_db_entry"
        # Action body is a pyarrow Buffer, convert to bytes
        action_body_bytes = bytes(action.body)
        action_json = json.loads(action_body_bytes.decode("utf-8"))
        assert "createExperimentDbEntry" in action_json
        assert action_json["createExperimentDbEntry"]["spaceId"] == "test_space"
        assert (
            action_json["createExperimentDbEntry"]["datasetId"] == "dataset_123"
        )
        assert (
            action_json["createExperimentDbEntry"]["experimentName"]
            == "test_experiment"
        )

    @patch("arize._flight.client.ArizeFlightClient.do_action")
    def test_init_experiment_request_fields(
        self,
        mock_do_action: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test that request contains all required fields."""
        # Setup mocks
        mock_result = Mock()
        response = flight_pb2.CreateExperimentDBEntryResponse()
        response.experiment_id = "exp_123"
        response.trace_model_name = "trace_model_456"
        mock_result.body.to_pybytes.return_value = response.SerializeToString()

        mock_do_action.return_value = iter([mock_result])

        # Execute
        flight_client.init_experiment(
            space_id="space_abc",
            dataset_id="dataset_xyz",
            experiment_name="my_experiment",
        )

        # Verify all fields are present
        call_args = mock_do_action.call_args
        action = call_args[0][0]
        # Action body is a pyarrow Buffer, convert to bytes
        action_body_bytes = bytes(action.body)
        action_json = json.loads(action_body_bytes.decode("utf-8"))

        entry = action_json["createExperimentDbEntry"]
        assert entry["spaceId"] == "space_abc"
        assert entry["datasetId"] == "dataset_xyz"
        assert entry["experimentName"] == "my_experiment"

    @patch("arize._flight.client.ArizeFlightClient.do_action")
    def test_init_experiment_none_response(
        self,
        mock_do_action: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test handling of None response (empty iterator)."""
        mock_do_action.return_value = iter([])

        # Execute
        result = flight_client.init_experiment(
            space_id="test_space",
            dataset_id="dataset_123",
            experiment_name="test_experiment",
        )

        # Verify
        assert result is None

    @patch("arize._flight.client.ArizeFlightClient.do_action")
    def test_init_experiment_flight_exception(
        self,
        mock_do_action: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test that Flight exceptions are wrapped in RuntimeError."""
        mock_do_action.side_effect = Exception("Flight connection failed")

        # Execute & Verify
        with pytest.raises(
            RuntimeError, match="Failed to init experiment test_experiment"
        ):
            flight_client.init_experiment(
                space_id="test_space",
                dataset_id="dataset_123",
                experiment_name="test_experiment",
            )

    @patch("arize._flight.client.ArizeFlightClient.do_action")
    def test_init_experiment_protobuf_parsing(
        self,
        mock_do_action: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test that protobuf response is correctly parsed."""
        # Setup mocks with specific values
        mock_result = Mock()
        response = flight_pb2.CreateExperimentDBEntryResponse()
        response.experiment_id = "unique_exp_id_789"
        response.trace_model_name = "unique_trace_model_123"
        mock_result.body.to_pybytes.return_value = response.SerializeToString()

        mock_do_action.return_value = iter([mock_result])

        # Execute
        result = flight_client.init_experiment(
            space_id="test_space",
            dataset_id="dataset_123",
            experiment_name="test_experiment",
        )

        # Verify parsing extracted correct values
        assert result is not None
        assert result[0] == "unique_exp_id_789"
        assert result[1] == "unique_trace_model_123"
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch("arize._flight.client.ArizeFlightClient.do_action")
    def test_init_experiment_returns_tuple(
        self,
        mock_do_action: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Test that return value is a tuple of (experiment_id, trace_model_name)."""
        # Setup mocks
        mock_result = Mock()
        response = flight_pb2.CreateExperimentDBEntryResponse()
        response.experiment_id = "exp_abc"
        response.trace_model_name = "trace_xyz"
        mock_result.body.to_pybytes.return_value = response.SerializeToString()

        mock_do_action.return_value = iter([mock_result])

        # Execute
        result = flight_client.init_experiment(
            space_id="test_space",
            dataset_id="dataset_123",
            experiment_name="test_experiment",
        )

        # Verify type and structure
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(item, str) for item in result)
