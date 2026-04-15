r"""Integration tests for ResourceRestrictionsClient against the real Arize API.

These tests are skipped unless the required environment variables are set.
They verify the full restrict/unrestrict round-trip against a real project.

Required environment variables:
    ARIZE_API_KEY: Arize API key with PROJECT_RESTRICT permission.
    ARIZE_TEST_PROJECT_ID: Global ID of an existing project to use as the test target.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_PROJECT_ID=<project_id> \
        pytest tests/integration/test_resource_restrictions.py -m integration -v
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arize.client import ArizeClient

import pytest

API_KEY = os.environ.get("ARIZE_API_KEY", "")
PROJECT_ID = os.environ.get("ARIZE_TEST_PROJECT_ID", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY or not PROJECT_ID,
        reason="ARIZE_API_KEY and ARIZE_TEST_PROJECT_ID must be set",
    ),
]


@pytest.fixture(scope="module")
def arize_client() -> ArizeClient:
    """Create a real ArizeClient for integration tests."""
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


class TestResourceRestrictionsClient:
    """Integration tests for ResourceRestrictionsClient."""

    def test_restrict_returns_resource_restriction(self, arize_client) -> None:
        """restrict() should return a ResourceRestriction with the correct resource_id."""
        restriction = arize_client.resource_restrictions.restrict(
            resource_id=PROJECT_ID
        )
        assert restriction.resource_id == PROJECT_ID

    def test_restrict_is_idempotent(self, arize_client) -> None:
        """Calling restrict() twice on the same resource should not raise."""
        arize_client.resource_restrictions.restrict(resource_id=PROJECT_ID)
        restriction = arize_client.resource_restrictions.restrict(
            resource_id=PROJECT_ID
        )
        assert restriction.resource_id == PROJECT_ID

    def test_unrestrict_succeeds_after_restrict(self, arize_client) -> None:
        """unrestrict() should succeed (return None) when resource is restricted."""
        arize_client.resource_restrictions.restrict(resource_id=PROJECT_ID)
        result = arize_client.resource_restrictions.unrestrict(
            resource_id=PROJECT_ID
        )
        assert result is None

    def test_unrestrict_not_restricted_raises(self, arize_client) -> None:
        """unrestrict() should raise ApiException (404) when resource is not restricted."""
        from arize._generated.api_client.exceptions import ApiException

        # Ensure the resource is not restricted first
        with contextlib.suppress(Exception):
            arize_client.resource_restrictions.unrestrict(
                resource_id=PROJECT_ID
            )

        with pytest.raises(ApiException) as exc_info:
            arize_client.resource_restrictions.unrestrict(
                resource_id=PROJECT_ID
            )
        assert exc_info.value.status == 404

    def test_restrict_unrestrict_round_trip(self, arize_client) -> None:
        """Full round-trip: restrict, verify, then unrestrict."""
        restriction = arize_client.resource_restrictions.restrict(
            resource_id=PROJECT_ID
        )
        assert restriction.resource_id == PROJECT_ID

        arize_client.resource_restrictions.unrestrict(resource_id=PROJECT_ID)

        # After unrestricting, a second unrestrict should fail with 404
        from arize._generated.api_client.exceptions import ApiException

        with pytest.raises(ApiException) as exc_info:
            arize_client.resource_restrictions.unrestrict(
                resource_id=PROJECT_ID
            )
        assert exc_info.value.status == 404
