"""Integration tests for TracesClient.list() against the real Arize API.

Run with:
    ARIZE_API_KEY=<key> ARIZE_SPACE_ID=<space-id> ARIZE_TEST_PROJECT_NAME=<project> \
        pytest tests/integration/test_traces_flows.py -m integration -v
"""

from __future__ import annotations

import os
from typing import Any

import pytest

API_KEY = os.environ.get("ARIZE_API_KEY", "")
SPACE_ID = os.environ.get("ARIZE_SPACE_ID", "")
PROJECT_NAME = os.environ.get("ARIZE_TEST_PROJECT_NAME", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY or not SPACE_ID or not PROJECT_NAME,
        reason=(
            "ARIZE_API_KEY, ARIZE_SPACE_ID, and ARIZE_TEST_PROJECT_NAME "
            "must be set"
        ),
    ),
]


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def traces_client(arize_client: Any) -> Any:
    return arize_client.traces


class TestTracesList:
    """End-to-end flows for TracesClient.list()."""

    def test_list_returns_traces_and_pagination(
        self, traces_client: Any
    ) -> None:
        """list() should return a response with .traces and .pagination."""
        resp = traces_client.list(
            project=PROJECT_NAME,
            space=SPACE_ID,
            limit=10,
        )

        assert hasattr(resp, "traces")
        assert hasattr(resp, "pagination")
        assert isinstance(resp.traces, list)

    def test_list_with_filter(self, traces_client: Any) -> None:
        """list() should accept a filter expression without error."""
        resp = traces_client.list(
            project=PROJECT_NAME,
            space=SPACE_ID,
            filter="status_code = 'ERROR'",
            limit=5,
        )

        assert hasattr(resp, "traces")
        assert hasattr(resp, "pagination")
