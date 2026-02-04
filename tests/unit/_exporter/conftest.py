"""Exporter module-specific fixtures."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import pyarrow as pa
import pytest
from pyarrow import flight

from arize._exporter.client import ArizeExportClient
from arize.ml.types import Environments


@pytest.fixture
def mock_flight_client() -> Mock:
    """Create a mock PyArrow FlightClient for exporter tests.

    Note: This mocks pyarrow.flight.FlightClient, not ArizeFlightClient.
    """
    return Mock(spec=flight.FlightClient)


@pytest.fixture
def export_client(mock_flight_client: Mock) -> ArizeExportClient:
    """Create an ArizeExportClient with mocked FlightClient."""
    return ArizeExportClient(flight_client=mock_flight_client)


@pytest.fixture
def valid_export_params() -> dict:
    """Create valid export parameters."""
    return {
        "space_id": "space123",
        "model_id": "model456",
        "environment": Environments.PRODUCTION,
        "start_time": datetime(2024, 1, 1),
        "end_time": datetime(2024, 1, 2),
    }


@pytest.fixture
def sample_arrow_table() -> pa.Table:
    """Create a sample PyArrow table for exporter tests."""
    return pa.table(
        {
            "time": [1000, 2000, 3000],
            "value": [1.0, 2.0, 3.0],
            "feature": ["a", "b", "c"],
            "null_col": [None, None, None],
        }
    )


@pytest.fixture
def sample_arrow_table_tracing() -> pa.Table:
    """Create a sample PyArrow table for tracing data."""
    return pa.table(
        {
            "time": [1000, 2000, 3000],
            "span.id": ["s1", "s2", "s3"],
            "attributes.llm.input_messages": [
                '{"role": "user"}',
                '{"role": "user"}',
                '{"role": "user"}',
            ],
        }
    )
