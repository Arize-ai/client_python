r"""Integration tests for RoleBindingsClient against the real Arize API.

These tests are skipped unless the required environment variables are set.
They verify the full create/get/update/delete round-trip for role bindings.

Required environment variables:
    ARIZE_API_KEY: Arize API key with ROLE_BINDING_CREATE/READ/DELETE permissions.
    ARIZE_TEST_USER_ID: Global ID of an existing user to bind roles to.
    ARIZE_TEST_ROLE_ID: Global ID of an existing role to bind.
    ARIZE_TEST_ROLE_ID_2: Global ID of a second role (used for update tests).
    ARIZE_TEST_PROJECT_ID: Global ID of an existing project to bind the role on.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_USER_ID=<uid> ARIZE_TEST_ROLE_ID=<rid> \
    ARIZE_TEST_ROLE_ID_2=<rid2> ARIZE_TEST_PROJECT_ID=<pid> \
        pytest tests/integration/test_role_bindings.py -m integration -v
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    from arize._generated.api_client import models
    from arize.client import ArizeClient

import pytest

API_KEY = os.environ.get("ARIZE_API_KEY", "")
USER_ID = os.environ.get("ARIZE_TEST_USER_ID", "")
ROLE_ID = os.environ.get("ARIZE_TEST_ROLE_ID", "")
ROLE_ID_2 = os.environ.get("ARIZE_TEST_ROLE_ID_2", "")
PROJECT_ID = os.environ.get("ARIZE_TEST_PROJECT_ID", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY or not USER_ID or not ROLE_ID or not PROJECT_ID,
        reason=(
            "ARIZE_API_KEY, ARIZE_TEST_USER_ID, ARIZE_TEST_ROLE_ID, and "
            "ARIZE_TEST_PROJECT_ID must be set"
        ),
    ),
]


@pytest.fixture(scope="module")
def arize_client() -> ArizeClient:
    """Create a real ArizeClient for integration tests."""
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def role_binding(
    arize_client: ArizeClient,
) -> Generator[models.RoleBinding | None, None, None]:
    """Create a role binding for the test session and clean up afterwards."""
    from arize._generated.api_client.models.role_binding_resource_type import (
        RoleBindingResourceType,
    )

    rb = arize_client.role_bindings.create(
        user_id=USER_ID,
        role_id=ROLE_ID,
        resource_type=RoleBindingResourceType.PROJECT,
        resource_id=PROJECT_ID,
    )
    yield rb
    # Cleanup: delete the binding after all tests in this module
    if rb is not None:
        with contextlib.suppress(Exception):
            arize_client.role_bindings.delete(binding_id=rb.id)


class TestRoleBindingsClientCreate:
    """Integration tests for RoleBindingsClient.create()."""

    def test_create_returns_role_binding(self, role_binding) -> None:
        """create() should return a RoleBinding with the expected fields."""
        assert role_binding.id
        assert role_binding.user_id == USER_ID
        assert role_binding.role_id == ROLE_ID
        assert role_binding.resource_id == PROJECT_ID

    def test_create_conflict_returns_none(
        self, arize_client, role_binding: models.RoleBinding | None
    ) -> None:
        """Creating a duplicate binding should silently return None."""
        from arize._generated.api_client.models.role_binding_resource_type import (
            RoleBindingResourceType,
        )

        result = arize_client.role_bindings.create(
            user_id=USER_ID,
            role_id=ROLE_ID,
            resource_type=RoleBindingResourceType.PROJECT,
            resource_id=PROJECT_ID,
        )
        assert result is None


class TestRoleBindingsClientGet:
    """Integration tests for RoleBindingsClient.get()."""

    def test_get_returns_matching_binding(
        self, arize_client, role_binding
    ) -> None:
        """get() should return the binding with the same ID."""
        fetched = arize_client.role_bindings.get(binding_id=role_binding.id)
        assert fetched.id == role_binding.id
        assert fetched.user_id == role_binding.user_id
        assert fetched.role_id == role_binding.role_id

    def test_get_not_found_raises(self, arize_client) -> None:
        """get() should raise ApiException (404) for a non-existent binding."""
        from arize._generated.api_client.exceptions import ApiException

        with pytest.raises(ApiException) as exc_info:
            arize_client.role_bindings.get(
                binding_id="Um9sZUJpbmRpbmc6OTk5OTk5OTk5OTk="  # unlikely to exist
            )
        assert exc_info.value.status == 404


@pytest.mark.skipif(
    not ROLE_ID_2,
    reason="ARIZE_TEST_ROLE_ID_2 must be set for update tests",
)
class TestRoleBindingsClientUpdate:
    """Integration tests for RoleBindingsClient.update()."""

    def test_update_changes_role(self, arize_client, role_binding) -> None:
        """update() should replace the role on the binding."""
        updated = arize_client.role_bindings.update(
            binding_id=role_binding.id,
            role_id=ROLE_ID_2,
        )
        assert updated.id == role_binding.id
        assert updated.role_id == ROLE_ID_2

        # Restore original role for subsequent tests
        arize_client.role_bindings.update(
            binding_id=role_binding.id,
            role_id=ROLE_ID,
        )


class TestRoleBindingsClientDelete:
    """Integration tests for RoleBindingsClient.delete()."""

    def test_delete_returns_none(self, arize_client) -> None:
        """delete() should return None and remove the binding."""
        from arize._generated.api_client.exceptions import ApiException
        from arize._generated.api_client.models.role_binding_resource_type import (
            RoleBindingResourceType,
        )

        # Create a temporary binding to delete
        rb = arize_client.role_bindings.create(
            user_id=USER_ID,
            role_id=ROLE_ID,
            resource_type=RoleBindingResourceType.PROJECT,
            resource_id=PROJECT_ID,
        )

        result = arize_client.role_bindings.delete(binding_id=rb.id)
        assert result is None

        # Confirm it's gone
        with pytest.raises(ApiException) as exc_info:
            arize_client.role_bindings.get(binding_id=rb.id)
        assert exc_info.value.status == 404
