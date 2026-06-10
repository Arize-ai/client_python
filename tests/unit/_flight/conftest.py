"""Flight module-specific fixtures."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from arize._flight.client import ArizeFlightClient
from arize.config import SDKConfiguration


@pytest.fixture
def flight_client() -> ArizeFlightClient:
    """Create a test ArizeFlightClient instance."""
    return ArizeFlightClient(
        sdk_config=SDKConfiguration(
            api_key="test_api_key",
            flight_host="test-host.com",
            flight_port=443,
            flight_scheme="https",
            pyarrow_max_chunksize=1000,
            request_verify=True,
        )
    )


@pytest.fixture
def flight_client_localhost() -> ArizeFlightClient:
    """Create a test ArizeFlightClient instance for localhost."""
    return ArizeFlightClient(
        sdk_config=SDKConfiguration(
            api_key="test_api_key",
            flight_host="localhost",
            flight_port=8080,
            flight_scheme="http",
            pyarrow_max_chunksize=1000,
            request_verify=True,
        )
    )


@pytest.fixture
def sample_pa_table() -> pa.Table:
    """Create a PyArrow table with Flight-specific columns."""
    return pa.table(
        {
            "span_id": ["span1", "span2", "span3"],
            "value": [1.0, 2.0, 3.0],
            "label": ["good", "bad", "good"],
        }
    )


@pytest.fixture
def sample_pa_table_large() -> pa.Table:
    """Create a large PyArrow table for Flight chunking tests."""
    return pa.table(
        {
            "id": [f"id_{i}" for i in range(100)],
            "value": [float(i) for i in range(100)],
        }
    )


@pytest.fixture
def sample_dataset_df() -> pd.DataFrame:
    """Create sample dataset DataFrame for Flight tests."""
    return pd.DataFrame(
        {
            "id": ["ex1", "ex2", "ex3"],
            "input": [
                '{"query": "test1"}',
                '{"query": "test2"}',
                '{"query": "test3"}',
            ],
            "output": [
                '{"response": "answer1"}',
                '{"response": "answer2"}',
                '{"response": "answer3"}',
            ],
        }
    )


@pytest.fixture
def sample_experiment_df() -> pd.DataFrame:
    """Create sample experiment DataFrame for Flight tests."""
    return pd.DataFrame(
        {
            "run_id": ["run1", "run2"],
            "config": ['{"param": "value1"}', '{"param": "value2"}'],
            "metrics": ['{"accuracy": 0.95}', '{"accuracy": 0.92}'],
        }
    )
