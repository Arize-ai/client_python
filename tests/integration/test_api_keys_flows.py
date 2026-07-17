"""Integration tests for ApiKeysClient end-to-end flows against the real Arize API.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_ID=<space> \
        pytest tests/integration/test_api_keys_flows.py -m integration -v
"""

from __future__ import annotations

import contextlib
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

API_KEY = os.environ.get("ARIZE_API_KEY", "")
SPACE_ID = os.environ.get("ARIZE_TEST_SPACE_ID", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY,
        reason="ARIZE_API_KEY must be set",
    ),
    pytest.mark.skipif(
        not SPACE_ID,
        reason="ARIZE_TEST_SPACE_ID must be set",
    ),
]


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def api_keys_client(arize_client) -> Any:
    return arize_client.api_keys


class TestApiKeysCRUD:
    """End-to-end CRUD flows for ApiKeysClient."""

    def test_create_service_key(self, api_keys_client) -> None:
        """Create a service key and verify the raw key is returned."""
        name = _unique("sdk-test-svc-key")
        created = api_keys_client.create_service_key(
            name=name,
            space=SPACE_ID,
            description="SDK integration test key",
        )
        try:
            assert created.key is not None
            assert len(created.key) > 0
            assert created.id is not None
        finally:
            api_keys_client.revoke(api_key_id=created.id)

    def test_create_and_list_service_key(self, api_keys_client) -> None:
        """Created service key appears in list() results."""
        name = _unique("sdk-test-svc-key")
        created = api_keys_client.create_service_key(
            name=name,
            space=SPACE_ID,
        )
        try:
            resp = api_keys_client.list(
                key_type="SERVICE", space=SPACE_ID, limit=100
            )
            key_ids = [k.id for k in resp.api_keys]
            assert created.id in key_ids
        finally:
            api_keys_client.revoke(api_key_id=created.id)

    def test_refresh_service_key_default_invalidation(
        self, api_keys_client
    ) -> None:
        """Refresh without grace period should replace the key immediately."""
        created = api_keys_client.create_service_key(
            name=_unique("sdk-test-svc-key-refresh"),
            space=SPACE_ID,
        )
        refreshed_id: str | None = None
        try:
            refreshed = api_keys_client.refresh(api_key_id=created.id)
            refreshed_id = refreshed.id

            assert refreshed.key is not None
            assert len(refreshed.key) > 0
            assert refreshed.id != created.id
            assert refreshed.name == created.name

            active_resp = api_keys_client.list(
                key_type="SERVICE", space=SPACE_ID, status="ACTIVE", limit=100
            )
            active_ids = [k.id for k in active_resp.api_keys]
            assert refreshed.id in active_ids
            assert created.id not in active_ids
        finally:
            if refreshed_id is not None:
                with contextlib.suppress(Exception):
                    api_keys_client.revoke(api_key_id=refreshed_id)
            with contextlib.suppress(Exception):
                api_keys_client.revoke(api_key_id=created.id)

    def test_refresh_service_key_with_grace_period(
        self, api_keys_client
    ) -> None:
        """Refresh with grace period keeps old key active with bounded expiry."""
        created = api_keys_client.create_service_key(
            name=_unique("sdk-test-svc-key-grace"),
            space=SPACE_ID,
        )
        refreshed_id: str | None = None
        grace_seconds = 300
        lower_bound = datetime.now(timezone.utc) + timedelta(
            seconds=grace_seconds - 30
        )
        upper_bound = datetime.now(timezone.utc) + timedelta(
            seconds=grace_seconds + 60
        )

        try:
            refreshed = api_keys_client.refresh(
                api_key_id=created.id,
                grace_period_seconds=grace_seconds,
            )
            refreshed_id = refreshed.id

            assert refreshed.key is not None
            assert len(refreshed.key) > 0
            assert refreshed.id != created.id

            active_resp = api_keys_client.list(
                key_type="SERVICE", space=SPACE_ID, status="ACTIVE", limit=100
            )
            active_by_id = {k.id: k for k in active_resp.api_keys}
            assert refreshed.id in active_by_id
            assert created.id in active_by_id

            old_key = active_by_id[created.id]
            assert old_key.expires_at is not None
            assert old_key.expires_at >= lower_bound
            assert old_key.expires_at <= upper_bound
        finally:
            if refreshed_id is not None:
                with contextlib.suppress(Exception):
                    api_keys_client.revoke(api_key_id=refreshed_id)
            with contextlib.suppress(Exception):
                api_keys_client.revoke(api_key_id=created.id)
