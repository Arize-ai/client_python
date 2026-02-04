"""Shared test fixtures for all test suites.

This module provides common fixtures used across unit and integration tests.
It helps maintain DRY (Don't Repeat Yourself) principles by centralizing
commonly used test data and mock objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock

import pyarrow as pa
import pytest

from arize.config import SDKConfiguration

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


@pytest.fixture
def test_api_key() -> str:
    """Provide mock API key for testing.

    Returns:
        str: A test API key to use in unit tests.
    """
    return "test_api_key_12345"


@pytest.fixture
def mock_sdk_config() -> Mock:
    """Provide mock SDKConfiguration for testing.

    Returns:
        Mock: A mock SDKConfiguration with common attributes pre-configured.
    """
    config = Mock(spec=SDKConfiguration)
    config.api_key = "test_api_key_12345"
    config.api_url = "https://api.arize.com"
    config.cache_dir = "~/.arize/cache"
    return config


@pytest.fixture
def sample_arrow_table() -> pa.Table:
    """Provide sample Arrow table for testing.

    Returns:
        pa.Table: A small Arrow table with 3 rows for basic testing.
    """
    return pa.table(
        {
            "id": ["id1", "id2", "id3"],
            "value": [1.0, 2.0, 3.0],
            "label": ["a", "b", "c"],
        }
    )


@pytest.fixture
def sample_arrow_table_large() -> pa.Table:
    """Provide large Arrow table for chunking tests.

    Returns:
        pa.Table: A large Arrow table with 1000 rows for testing
                 chunking, pagination, and performance scenarios.
    """
    return pa.table(
        {
            "id": [f"id_{i}" for i in range(1000)],
            "value": [float(i) for i in range(1000)],
        }
    )


@pytest.fixture
def tmp_parquet_path(tmp_path: Path) -> Path:
    """Provide temporary path for parquet exports.

    Args:
        tmp_path: pytest's built-in tmp_path fixture.

    Returns:
        Path: Temporary path for parquet file exports.
    """
    return tmp_path / "export.parquet"


@pytest.fixture
def mock_flight_client() -> MagicMock:
    """Provide pre-configured mock Flight client.

    Returns:
        MagicMock: A mock ArizeFlightClient with common methods pre-configured.
    """
    mock_client = MagicMock()
    mock_client.api_key = "test_api_key"
    mock_client.host = "test-host.com"
    mock_client.port = 443
    mock_client.scheme = "https"
    mock_client.max_chunksize = 1000
    mock_client.request_verify = True
    return mock_client


@pytest.fixture
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set up the required ARIZE_API_KEY environment variable.

    Args:
        monkeypatch: pytest's monkeypatch fixture.

    Returns:
        str: The test API key that was set in the environment.
    """
    monkeypatch.setenv("ARIZE_API_KEY", "test_key")
    return "test_key"


@pytest.fixture
def config_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., SDKConfiguration]:
    """Factory fixture for creating SDKConfiguration with API key.

    This fixture automatically sets up the required ARIZE_API_KEY environment
    variable and provides a factory function for creating SDKConfiguration
    instances with custom parameters.

    Args:
        monkeypatch: pytest's monkeypatch fixture.

    Returns:
        Callable: Factory function that creates SDKConfiguration instances.
    """
    monkeypatch.setenv("ARIZE_API_KEY", "test_key")

    def _create(**kwargs: object) -> SDKConfiguration:
        return SDKConfiguration(**kwargs)

    return _create


@pytest.fixture
def mock_arize_client(test_api_key: str) -> Iterator[Any]:
    """Provide a mocked ArizeClient for testing.

    This fixture sets up an ArizeClient with mocked SDKConfiguration,
    eliminating the need for repeated mock setup in tests.

    Args:
        test_api_key: The test API key fixture.

    Returns:
        ArizeClient: An ArizeClient instance with mocked configuration.
    """
    from unittest.mock import patch

    from arize.client import ArizeClient

    with patch("arize.client.SDKConfiguration") as mock_config_cls:
        mock_config_cls.return_value = Mock()
        yield ArizeClient(api_key=test_api_key)


@pytest.fixture
def mock_reader_factory() -> Callable[..., tuple[Mock, int]]:
    """Factory for creating mock readers with arrow tables.

    This fixture provides a factory function that creates mock PyArrow
    RecordBatchReader objects for testing data reading scenarios.

    Returns:
        Callable: Factory function that creates mock readers.
    """

    def _create_mock_reader(
        chunks: list[pa.Table] | None = None, total_records: int = 0
    ) -> tuple[Mock, int]:
        """Create a mock reader with specified chunks.

        Args:
            chunks: List of PyArrow tables to return as chunks.
            total_records: Total number of records across all chunks.

        Returns:
            tuple: (mock_reader, total_records)
        """
        if chunks is None:
            chunks = []

        mock_reader = Mock()
        mock_batches = [Mock() for _ in chunks]
        for batch, chunk in zip(mock_batches, chunks, strict=True):
            batch.data = chunk
        mock_reader.read_chunk.side_effect = [*mock_batches, StopIteration()]

        return mock_reader, total_records

    return _create_mock_reader


@pytest.fixture
def mock_flight_info_factory() -> Callable[[int], Mock]:
    """Factory for creating mock flight info objects.

    This fixture provides a factory function that creates mock FlightInfo
    objects for testing Arrow Flight interactions.

    Returns:
        Callable: Factory function that creates mock FlightInfo instances.
    """

    def _create_mock_flight_info(total_records: int = 10) -> Mock:
        """Create a mock FlightInfo with specified total records.

        Args:
            total_records: Total number of records in the flight.

        Returns:
            Mock: A mock FlightInfo object.
        """
        mock_flight_info = Mock()
        mock_flight_info.total_records = total_records
        mock_endpoint = Mock()
        mock_endpoint.ticket = Mock()
        mock_flight_info.endpoints = [mock_endpoint]
        return mock_flight_info

    return _create_mock_flight_info
