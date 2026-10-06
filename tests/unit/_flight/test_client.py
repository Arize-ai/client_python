"""Unit tests for arize._flight.client module.

This file contains all tests for ArizeFlightClient:
- Low-level internals (initialization, connection, properties, passthrough methods)
- High-level workflows (log_arrow_table, create_dataset, get_dataset_examples, etc.)

All tests use mocks and are marked with @pytest.mark.unit.
"""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyarrow import flight

from arize._flight.client import (
    ArizeFlightClient,
    _get_pb_flight_doput_request,
    append_to_pyarrow_metadata,
)
from arize._flight.types import FlightRequestType
from arize._generated.protocol.flight import flight_pb2
from arize.config import SDKConfiguration
from arize.datasets.upload import (
    flight_schema,
    iter_flight_batches,
    source_type,
)
from arize.exceptions.arrow import RecordBatchTooLargeError
from arize.utils.arrow import split_batches_by_byte_budget
from arize.utils.file_sources import open_source, unified_source_schema
from arize.utils.openinference_conversion import convert_json_str_to_dict

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


# ==================== Helper Functions ====================


def _reject_over(
    limit: int, error: Exception
) -> Callable[[pa.RecordBatch], None]:
    """Build a write_batch side effect that fails like the server on a large batch."""

    def write_batch(batch: pa.RecordBatch) -> None:
        if batch.nbytes > limit:
            raise error

    return write_batch


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
        """Test client initialization reads connection settings from sdk_config."""
        config = SDKConfiguration(
            api_key="test_key",
            flight_host="example.com",
            flight_port=443,
            flight_scheme="https",
            pyarrow_max_chunksize=2000,
            request_verify=False,
        )
        client = ArizeFlightClient(sdk_config=config)
        assert client.sdk_config is config
        assert client.sdk_config.flight_host == "example.com"
        assert client.sdk_config.flight_port == 443
        assert client.sdk_config.flight_scheme == "https"
        assert client.sdk_config.pyarrow_max_chunksize == 2000
        assert client.sdk_config.request_verify is False

    def test_init_internal_client_is_none(
        self, flight_client: ArizeFlightClient
    ) -> None:
        """Test that internal _client is None on initialization."""
        assert object.__getattribute__(flight_client, "_client") is None

    def test_frozen_dataclass(self, flight_client: ArizeFlightClient) -> None:
        """Test that ArizeFlightClient is frozen."""
        with pytest.raises(AttributeError):
            flight_client.sdk_config = None  # type: ignore[misc, assignment]


@pytest.mark.unit
class TestArizeFlightClientProperties:
    """Test ArizeFlightClient properties."""

    def test_headers_property(self, flight_client: ArizeFlightClient) -> None:
        """Test that headers property returns correct headers."""
        headers = flight_client.headers
        assert isinstance(headers, list)
        assert len(headers) == 6
        assert (b"origin", b"arize-logging-client") in headers
        assert (b"auth-token-bin", b"test_api_key") in headers
        assert (b"sdk-language", b"python") in headers
        assert (b"sdk-package-name", b"arize") in headers

    def test_headers_contain_version_info(
        self, flight_client: ArizeFlightClient
    ) -> None:
        """Test that headers contain version information."""
        headers = flight_client.headers
        headers_dict = dict(headers)
        assert b"language-version" in headers_dict
        assert b"sdk-version" in headers_dict

    def test_headers_include_default_headers(self) -> None:
        """Test that user default_headers are byte-encoded into Flight headers."""
        client = ArizeFlightClient(
            sdk_config=SDKConfiguration(
                api_key="test_api_key",
                default_headers={"x-tenant": "acme"},
            )
        )
        headers_dict = dict(client.headers)
        assert headers_dict[b"x-tenant"] == b"acme"
        # Built-in Flight headers are still present.
        assert headers_dict[b"origin"] == b"arize-logging-client"
        assert headers_dict[b"auth-token-bin"] == b"test_api_key"

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
            sdk_config=SDKConfiguration(
                api_key="test_key",
                flight_host="example.com",
                flight_port=443,
                flight_scheme="https",
                pyarrow_max_chunksize=1000,
                request_verify=False,
            )
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
            reader=sample_pa_table.to_reader(),
            **extra_kwargs,
        )

        # Verify
        assert isinstance(result, response_class)
        assert getattr(result, response_field) == expected_value
        mock_writer.write_batch.assert_called_once()
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
                reader=sample_pa_table.to_reader(),
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
            reader=sample_pa_table.to_reader(),
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
                reader=sample_pa_table.to_reader(),
                project_name="test_project",
            )

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    @patch("arize._flight.client.get_pb_schema_tracing")
    def test_log_arrow_table_writes_every_batch(
        self,
        mock_get_schema: Mock,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table_large: pa.Table,
    ) -> None:
        """Test that each batch is written before done_writing."""
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
            reader=sample_pa_table_large.to_reader(max_chunksize=30),
            project_name="test_project",
        )

        written = [
            call.args[0]
            for call in mock_writer.method_calls
            if call[0] == "write_batch"
        ]
        assert [b.num_rows for b in written] == [30, 30, 30, 10]
        method_names = [call[0] for call in mock_writer.method_calls]
        assert method_names.index("done_writing") > max(
            i for i, n in enumerate(method_names) if n == "write_batch"
        )

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
            reader=sample_pa_table.to_reader(),
            project_name="test_project",
        )

        # Verify metadata was appended
        mock_append_metadata.assert_called_once()
        call_args = mock_append_metadata.call_args[0]
        assert call_args[0] == sample_pa_table.schema

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    @patch("arize._flight.client.get_pb_schema_tracing")
    def test_log_arrow_table_sends_oversized_row_for_server_to_reject(
        self,
        mock_get_schema: Mock,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a row over the server limit is sent, so the server fails the stream."""
        monkeypatch.setattr(
            "arize._flight.client.FLIGHT_SERVER_MAX_MESSAGE_BYTES", 10_000
        )
        mock_schema = Mock()
        mock_schema.SerializeToString.return_value = b"schema_bytes"
        mock_get_schema.return_value = mock_schema
        mock_writer = create_context_mock_writer()
        server_error = flight.FlightServerError(
            "grpc: received message larger than max"
        )
        mock_writer.write_batch.side_effect = _reject_over(10_000, server_error)
        mock_do_put.return_value = (mock_writer, Mock())
        table = pa.table({"payload": ["x" * 100, "x" * 50_000]})

        with pytest.raises(RuntimeError) as excinfo:
            flight_client.log_arrow_table(
                space_id="test_space",
                request_type=FlightRequestType.EVALUATION,
                reader=table.to_reader(),
                project_name="test_project",
            )

        too_large = excinfo.value.__cause__
        assert isinstance(too_large, RecordBatchTooLargeError)
        assert too_large.__cause__ is server_error
        written = [c.args[0] for c in mock_writer.write_batch.call_args_list]
        assert written[-1].nbytes > 10_000
        mock_writer.done_writing.assert_not_called()

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    @patch("arize._flight.client.get_pb_schema_tracing")
    def test_log_arrow_table_respects_row_ceiling(
        self,
        mock_get_schema: Mock,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table_over_row_ceiling: pa.Table,
    ) -> None:
        """Test that incoming batches are re-cut to the configured row ceiling."""
        mock_schema = Mock()
        mock_schema.SerializeToString.return_value = b"schema_bytes"
        mock_get_schema.return_value = mock_schema
        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_response = Mock()
        mock_response.to_pybytes.return_value = (
            flight_pb2.WriteSpanEvaluationResponse().SerializeToString()
        )
        mock_metadata_reader.read.return_value = mock_response
        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        flight_client.log_arrow_table(
            space_id="test_space",
            request_type=FlightRequestType.EVALUATION,
            reader=sample_pa_table_over_row_ceiling.combine_chunks().to_reader(),
            project_name="test_project",
        )

        written = [
            call.args[0] for call in mock_writer.write_batch.call_args_list
        ]
        assert [batch.num_rows for batch in written] == [1000, 1000, 500]


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
            reader=sample_pa_table.to_reader(),
        )

        # Verify
        assert result == "dataset_12345"
        mock_writer.write_batch.assert_called_once()
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
            reader=sample_pa_table.to_reader(),
        )

        # Verify descriptor was created correctly
        call_args = mock_do_put.call_args
        descriptor = call_args[0][0]
        assert call_args[0][1] == sample_pa_table.schema

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
            reader=sample_pa_table.to_reader(),
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
                reader=sample_pa_table.to_reader(),
            )

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_write_workflow(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table_large: pa.Table,
    ) -> None:
        """Test the write workflow: every batch → done_writing → read."""
        # Setup mocks
        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_response = Mock()

        response = flight_pb2.CreateDatasetResponse()
        response.dataset_id = "dataset_123"
        mock_response.to_pybytes.return_value = response.SerializeToString()
        mock_metadata_reader.read.return_value = mock_response

        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        batches = sample_pa_table_large.to_batches(max_chunksize=30)
        assert len(batches) == 4

        # Execute
        flight_client.create_dataset(
            space_id="test_space",
            dataset_name="test_dataset",
            reader=pa.RecordBatchReader.from_batches(
                sample_pa_table_large.schema, iter(batches)
            ),
        )

        # Verify every batch was written, in order, before done_writing
        written = [
            call.args[0]
            for call in mock_writer.method_calls
            if call[0] == "write_batch"
        ]
        assert [b.num_rows for b in written] == [30, 30, 30, 10]
        assert mock_metadata_reader.read.called

        method_names = [call[0] for call in mock_writer.method_calls]
        assert method_names.index("done_writing") > max(
            i for i, n in enumerate(method_names) if n == "write_batch"
        )

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_batch_error_wrapped(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table: pa.Table,
    ) -> None:
        """Test that an error while producing batches is wrapped and stops the stream."""
        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        def failing_batches() -> Iterator[pa.RecordBatch]:
            yield sample_pa_table.to_batches()[0]
            raise ValueError("bad batch")

        with pytest.raises(RuntimeError) as excinfo:
            flight_client.create_dataset(
                space_id="test_space",
                dataset_name="test_dataset",
                reader=pa.RecordBatchReader.from_batches(
                    sample_pa_table.schema, failing_batches()
                ),
            )

        assert isinstance(excinfo.value.__cause__, ValueError)
        mock_writer.write_batch.assert_called_once()
        mock_writer.done_writing.assert_not_called()

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_sends_oversized_row_for_server_to_reject(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a row over the server limit is sent, so the server fails the stream."""
        monkeypatch.setattr(
            "arize._flight.client.FLIGHT_SERVER_MAX_MESSAGE_BYTES", 10_000
        )
        mock_writer = create_context_mock_writer()
        server_error = flight.FlightServerError(
            "grpc: received message larger than max"
        )
        mock_writer.write_batch.side_effect = _reject_over(10_000, server_error)
        mock_do_put.return_value = (mock_writer, Mock())
        table = pa.table({"payload": ["x" * 100, "x" * 50_000]})

        with pytest.raises(RuntimeError) as excinfo:
            flight_client.create_dataset(
                space_id="test_space",
                dataset_name="test_dataset",
                reader=table.to_reader(),
            )

        too_large = excinfo.value.__cause__
        assert isinstance(too_large, RecordBatchTooLargeError)
        assert too_large.__cause__ is server_error
        written = [c.args[0] for c in mock_writer.write_batch.call_args_list]
        assert written[-1].nbytes > 10_000
        mock_writer.done_writing.assert_not_called()

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_names_oversized_row_when_server_error_arrives_late(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that the rejection maps to RecordBatchTooLargeError on the response read."""
        monkeypatch.setattr(
            "arize._flight.client.FLIGHT_SERVER_MAX_MESSAGE_BYTES", 10_000
        )
        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        server_error = flight.FlightServerError(
            "grpc: received message larger than max"
        )
        mock_metadata_reader.read.side_effect = server_error
        mock_do_put.return_value = (mock_writer, mock_metadata_reader)
        table = pa.table({"payload": ["x" * 50_000]})

        with pytest.raises(RuntimeError) as excinfo:
            flight_client.create_dataset(
                space_id="test_space",
                dataset_name="test_dataset",
                reader=table.to_reader(),
            )

        too_large = excinfo.value.__cause__
        assert isinstance(too_large, RecordBatchTooLargeError)
        assert too_large.__cause__ is server_error

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_leaves_other_errors_unchanged(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table: pa.Table,
    ) -> None:
        """Test that a failure with every batch under the limit keeps its own cause."""
        mock_writer = create_context_mock_writer()
        server_error = flight.FlightUnavailableError("connection reset")
        mock_writer.write_batch.side_effect = server_error
        mock_do_put.return_value = (mock_writer, Mock())

        with pytest.raises(RuntimeError) as excinfo:
            flight_client.create_dataset(
                space_id="test_space",
                dataset_name="test_dataset",
                reader=sample_pa_table.to_reader(),
            )

        assert excinfo.value.__cause__ is server_error

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_respects_row_ceiling(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        sample_pa_table_over_row_ceiling: pa.Table,
    ) -> None:
        """Test that incoming batches are re-cut to the configured row ceiling."""
        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_response = Mock()
        mock_response.to_pybytes.return_value = (
            flight_pb2.CreateDatasetResponse(
                dataset_id="ds-1"
            ).SerializeToString()
        )
        mock_metadata_reader.read.return_value = mock_response
        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        flight_client.create_dataset(
            space_id="test_space",
            dataset_name="test_dataset",
            reader=sample_pa_table_over_row_ceiling.combine_chunks().to_reader(),
        )

        written = [
            call.args[0] for call in mock_writer.write_batch.call_args_list
        ]
        assert [batch.num_rows for batch in written] == [1000, 1000, 500]

    @patch("arize._flight.client.ArizeFlightClient.do_put")
    def test_create_dataset_bounds_file_batches_by_bytes(
        self,
        mock_do_put: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        """Test that batches streamed from a file arrive cut to the byte budget."""
        path = tmp_path / "wide.parquet"
        pq.write_table(
            pa.table(
                {
                    "id": [f"ex-{i}" for i in range(20)],
                    "text": ["x" * 1_000] * 20,
                }
            ),
            path,
        )
        sources = [open_source(path)]
        schema = flight_schema(unified_source_schema(sources, source_type))
        mock_writer = create_context_mock_writer()
        mock_metadata_reader = Mock()
        mock_response = Mock()
        mock_response.to_pybytes.return_value = (
            flight_pb2.CreateDatasetResponse(
                dataset_id="ds-1"
            ).SerializeToString()
        )
        mock_metadata_reader.read.return_value = mock_response
        mock_do_put.return_value = (mock_writer, mock_metadata_reader)

        with patch(
            "arize._flight.client.split_batches_by_byte_budget",
            partial(split_batches_by_byte_budget, max_batch_bytes=4_000),
        ):
            flight_client.create_dataset(
                space_id="test_space",
                dataset_name="test_dataset",
                reader=pa.RecordBatchReader.from_batches(
                    schema, iter_flight_batches(sources, schema, 20, 0)
                ),
            )

        written = [
            call.args[0] for call in mock_writer.write_batch.call_args_list
        ]
        assert len(written) > 1
        assert all(batch.nbytes <= 4_000 for batch in written)
        assert sum(batch.num_rows for batch in written) == 20


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
    def test_get_experiment_runs_preserves_output_strings(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
    ) -> None:
        """Keep task output strings raw while parsing other JSON columns."""
        mock_reader = Mock()
        mock_table = Mock()
        mock_table.to_pandas.return_value = pd.DataFrame(
            {
                "output": ['{"ok": true}'],
                "eval.test.metadata": ['{"source": "test"}'],
            }
        )
        mock_reader.read_all.return_value = mock_table
        mock_do_get.return_value = mock_reader

        result = flight_client.get_experiment_runs(
            space_id="test_space",
            experiment_id="exp_123",
        )

        assert result["output"].iloc[0] == '{"ok": true}'
        assert result["eval.test.metadata"].iloc[0] == {"source": "test"}

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


def _mock_stream_reader(
    schema: pa.Schema, batches: list[pa.RecordBatch]
) -> Mock:
    reader = Mock()
    reader.schema = schema
    chunks = [Mock(data=batch) for batch in batches]
    reader.read_chunk.side_effect = [*chunks, StopIteration()]
    reader.read_all.return_value = pa.Table.from_batches(batches, schema)
    return reader


def _do_get_ticket(mock_do_get: Mock) -> dict:
    return json.loads(mock_do_get.call_args[0][0].ticket.decode("utf-8"))


_EXPORT_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("input", pa.string()),
        ("output", pa.string()),
        ("eval.correctness.metadata", pa.string()),
    ]
)


def _export_batches() -> list[pa.RecordBatch]:
    return [
        pa.RecordBatch.from_pydict(
            {
                "id": ["a", "b"],
                "input": ["q1", "q2"],
                "output": ['{"answer": 1}', '{"answer": 2}'],
                "eval.correctness.metadata": ['{"k": "v1"}', '{"k": "v2"}'],
            },
            schema=_EXPORT_SCHEMA,
        ),
        pa.RecordBatch.from_pydict(
            {
                "id": ["c"],
                "input": ["q3"],
                "output": ['{"answer": 3}'],
                "eval.correctness.metadata": ['{"k": "v3"}'],
            },
            schema=_EXPORT_SCHEMA,
        ),
    ]


@pytest.mark.unit
class TestExportDatasetExamplesToParquet:
    """Test export_dataset_examples_to_parquet streaming workflows."""

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_writes_batches_in_order(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        batches = _export_batches()
        mock_do_get.return_value = _mock_stream_reader(_EXPORT_SCHEMA, batches)
        path = tmp_path / "examples.parquet"

        result = flight_client.export_dataset_examples_to_parquet(
            space_id="test_space",
            dataset_id="dataset_123",
            dataset_version_id=None,
            path=str(path),
        )

        assert result is None
        written = pq.read_table(path)
        assert written.equals(pa.Table.from_batches(batches, _EXPORT_SCHEMA))
        assert pq.ParquetFile(path).metadata.num_row_groups == 1
        mock_do_get.return_value.read_all.assert_not_called()
        assert list(tmp_path.iterdir()) == [path]

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_row_groups_bounded_by_byte_budget(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        batches = [*_export_batches(), *_export_batches()]
        mock_do_get.return_value = _mock_stream_reader(_EXPORT_SCHEMA, batches)
        path = tmp_path / "examples.parquet"
        budget = batches[0].nbytes + batches[1].nbytes

        with patch(
            "arize._flight.client.EXPORT_ROW_GROUP_BUDGET_BYTES", budget
        ):
            flight_client.export_dataset_examples_to_parquet(
                space_id="test_space",
                dataset_id="dataset_123",
                dataset_version_id=None,
                path=str(path),
            )

        metadata = pq.ParquetFile(path).metadata
        assert [
            metadata.row_group(i).num_rows
            for i in range(metadata.num_row_groups)
        ] == [3, 3]
        written = pq.read_table(path)
        assert written.equals(pa.Table.from_batches(batches, _EXPORT_SCHEMA))

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_ticket_matches_get_dataset_examples(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        mock_do_get.return_value = _mock_stream_reader(
            _EXPORT_SCHEMA, _export_batches()
        )
        flight_client.get_dataset_examples(
            space_id="test_space",
            dataset_id="dataset_123",
            dataset_version_id="v1",
        )
        get_ticket = _do_get_ticket(mock_do_get)

        mock_do_get.return_value = _mock_stream_reader(
            _EXPORT_SCHEMA, _export_batches()
        )
        flight_client.export_dataset_examples_to_parquet(
            space_id="test_space",
            dataset_id="dataset_123",
            dataset_version_id="v1",
            path=str(tmp_path / "examples.parquet"),
        )

        assert _do_get_ticket(mock_do_get) == get_ticket
        assert get_ticket["getDataset"]["datasetVersion"] == "v1"

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_empty_stream_writes_schema_only_file(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        mock_do_get.return_value = _mock_stream_reader(_EXPORT_SCHEMA, [])
        path = tmp_path / "examples.parquet"

        flight_client.export_dataset_examples_to_parquet(
            space_id="test_space",
            dataset_id="dataset_123",
            dataset_version_id=None,
            path=str(path),
        )

        written = pq.read_table(path)
        assert written.num_rows == 0
        assert written.schema.equals(_EXPORT_SCHEMA)

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_mid_stream_error_is_wrapped(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        reader = Mock()
        reader.schema = _EXPORT_SCHEMA
        reader.read_chunk.side_effect = [
            Mock(data=_export_batches()[0]),
            flight.FlightInternalError("stream broke"),
        ]
        mock_do_get.return_value = reader

        with pytest.raises(
            RuntimeError, match="Failed to export dataset id=dataset_123"
        ) as exc_info:
            flight_client.export_dataset_examples_to_parquet(
                space_id="test_space",
                dataset_id="dataset_123",
                dataset_version_id=None,
                path=str(tmp_path / "examples.parquet"),
            )
        assert isinstance(exc_info.value.__cause__, flight.FlightInternalError)
        assert list(tmp_path.iterdir()) == []

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_do_get_error_is_wrapped(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        mock_do_get.side_effect = Exception("Flight connection failed")
        path = tmp_path / "examples.parquet"

        with pytest.raises(
            RuntimeError,
            match="Failed to export dataset id=dataset_123: Flight connection failed",
        ):
            flight_client.export_dataset_examples_to_parquet(
                space_id="test_space",
                dataset_id="dataset_123",
                dataset_version_id=None,
                path=str(path),
            )
        assert not path.exists()

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_parity_with_get_dataset_examples(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        mock_do_get.return_value = _mock_stream_reader(
            _EXPORT_SCHEMA, _export_batches()
        )
        expected = flight_client.get_dataset_examples(
            space_id="test_space", dataset_id="dataset_123"
        )

        mock_do_get.return_value = _mock_stream_reader(
            _EXPORT_SCHEMA, _export_batches()
        )
        path = tmp_path / "examples.parquet"
        flight_client.export_dataset_examples_to_parquet(
            space_id="test_space",
            dataset_id="dataset_123",
            dataset_version_id=None,
            path=str(path),
        )

        exported = pq.read_table(path).to_pandas()
        assert exported["output"].iloc[0] == '{"answer": 1}'
        pd.testing.assert_frame_equal(
            convert_json_str_to_dict(exported), expected
        )
        assert expected["output"].iloc[0] == {"answer": 1}


@pytest.mark.unit
class TestExportExperimentRunsToParquet:
    """Test export_experiment_runs_to_parquet streaming workflows."""

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_writes_batches_in_order(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        batches = _export_batches()
        mock_do_get.return_value = _mock_stream_reader(_EXPORT_SCHEMA, batches)
        path = tmp_path / "runs.parquet"

        result = flight_client.export_experiment_runs_to_parquet(
            space_id="test_space",
            experiment_id="exp_123",
            path=str(path),
        )

        assert result is None
        written = pq.read_table(path)
        assert written.equals(pa.Table.from_batches(batches, _EXPORT_SCHEMA))
        mock_do_get.return_value.read_all.assert_not_called()

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_ticket_matches_get_experiment_runs(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        mock_do_get.return_value = _mock_stream_reader(
            _EXPORT_SCHEMA, _export_batches()
        )
        flight_client.get_experiment_runs(
            space_id="test_space", experiment_id="exp_123"
        )
        get_ticket = _do_get_ticket(mock_do_get)

        mock_do_get.return_value = _mock_stream_reader(
            _EXPORT_SCHEMA, _export_batches()
        )
        flight_client.export_experiment_runs_to_parquet(
            space_id="test_space",
            experiment_id="exp_123",
            path=str(tmp_path / "runs.parquet"),
        )

        assert _do_get_ticket(mock_do_get) == get_ticket
        assert get_ticket["getExperiment"]["experimentId"] == "exp_123"

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_empty_stream_writes_schema_only_file(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        mock_do_get.return_value = _mock_stream_reader(_EXPORT_SCHEMA, [])
        path = tmp_path / "runs.parquet"

        flight_client.export_experiment_runs_to_parquet(
            space_id="test_space",
            experiment_id="exp_123",
            path=str(path),
        )

        written = pq.read_table(path)
        assert written.num_rows == 0
        assert written.schema.equals(_EXPORT_SCHEMA)

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_mid_stream_error_is_wrapped(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        reader = Mock()
        reader.schema = _EXPORT_SCHEMA
        reader.read_chunk.side_effect = [
            Mock(data=_export_batches()[0]),
            flight.FlightInternalError("stream broke"),
        ]
        mock_do_get.return_value = reader
        path = tmp_path / "runs.parquet"
        path.write_bytes(b"previous export")

        with pytest.raises(
            RuntimeError, match="Failed to export experiment id=exp_123"
        ):
            flight_client.export_experiment_runs_to_parquet(
                space_id="test_space",
                experiment_id="exp_123",
                path=str(path),
            )

        assert list(tmp_path.iterdir()) == [path]
        assert path.read_bytes() == b"previous export"

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_do_get_error_is_wrapped(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        mock_do_get.side_effect = Exception("Flight connection failed")

        with pytest.raises(
            RuntimeError,
            match="Failed to export experiment id=exp_123: Flight connection failed",
        ):
            flight_client.export_experiment_runs_to_parquet(
                space_id="test_space",
                experiment_id="exp_123",
                path=str(tmp_path / "runs.parquet"),
            )

    @patch("arize._flight.client.ArizeFlightClient.do_get")
    def test_parity_with_get_experiment_runs(
        self,
        mock_do_get: Mock,
        flight_client: ArizeFlightClient,
        tmp_path: Path,
    ) -> None:
        mock_do_get.return_value = _mock_stream_reader(
            _EXPORT_SCHEMA, _export_batches()
        )
        expected = flight_client.get_experiment_runs(
            space_id="test_space", experiment_id="exp_123"
        )

        mock_do_get.return_value = _mock_stream_reader(
            _EXPORT_SCHEMA, _export_batches()
        )
        path = tmp_path / "runs.parquet"
        flight_client.export_experiment_runs_to_parquet(
            space_id="test_space",
            experiment_id="exp_123",
            path=str(path),
        )

        exported = convert_json_str_to_dict(
            pq.read_table(path).to_pandas(),
            excluded_columns=("result", "output"),
        )
        pd.testing.assert_frame_equal(exported, expected)
        assert all(isinstance(v, str) for v in exported["output"])
        assert exported["eval.correctness.metadata"].iloc[0] == {"k": "v1"}


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
