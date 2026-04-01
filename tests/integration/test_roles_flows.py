"""Integration tests for RolesClient end-to-end flows against the real Arize API.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_roles_flows.py -m integration -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from arize.utils.resolve import is_resource_id

API_KEY = os.environ.get("ARIZE_API_KEY", "")
SPACE_NAME = os.environ.get("ARIZE_TEST_SPACE_NAME", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY or not SPACE_NAME,
        reason="ARIZE_API_KEY and ARIZE_TEST_SPACE_NAME must be set",
    ),
]


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def roles_client(arize_client) -> Any:
    return arize_client.roles


class TestRolesCRUD:
    """End-to-end CRUD flows for RolesClient."""

    def test_create_get_delete_by_id(self, roles_client) -> None:
        """Create a role, retrieve it by ID, then delete it."""
        from arize._generated.api_client.models.permission import Permission

        name = _unique("sdk-test-role")
        role = roles_client.create(
            name=name,
            permissions=[Permission.PROJECT_READ],
            description="Created by SDK integration test",
        )
        try:
            assert role.name == name
            assert is_resource_id(role.id)

            fetched = roles_client.get(role=role.id)
            assert fetched.id == role.id
            assert fetched.name == name
        finally:
            roles_client.delete(role=role.id)

    def test_create_get_delete_by_name(self, roles_client) -> None:
        """Create a role, retrieve and delete it using its name."""
        from arize._generated.api_client.models.permission import Permission

        name = _unique("sdk-test-role")
        role = roles_client.create(
            name=name,
            permissions=[Permission.PROJECT_READ, Permission.DATASET_CREATE],
        )
        try:
            fetched = roles_client.get(role=name)
            assert fetched.id == role.id
        finally:
            roles_client.delete(role=name)

    def test_create_update_delete(self, roles_client) -> None:
        """Create a role, update its name and description, then delete it."""
        from arize._generated.api_client.models.permission import Permission

        original_name = _unique("sdk-test-role")
        role = roles_client.create(
            name=original_name,
            permissions=[Permission.PROJECT_READ],
        )
        updated_name = _unique("sdk-test-role-upd")
        try:
            updated = roles_client.update(
                role_id=role.id,
                name=updated_name,
                description="Updated by SDK integration test",
            )
            assert updated.id == role.id
            assert updated.name == updated_name

            fetched = roles_client.get(role=role.id)
            assert fetched.name == updated_name
        finally:
            roles_client.delete(role=role.id)

    def test_create_appears_in_list(self, roles_client) -> None:
        """Newly created role appears in list() results."""
        from arize._generated.api_client.models.permission import Permission

        name = _unique("sdk-test-role")
        role = roles_client.create(
            name=name,
            permissions=[Permission.PROJECT_READ],
        )
        try:
            resp = roles_client.list(is_predefined=False, limit=100)
            role_ids = [r.id for r in resp.roles]
            assert role.id in role_ids
        finally:
            roles_client.delete(role=role.id)

    def test_update_permissions(self, roles_client) -> None:
        """Update a role's permissions and confirm the change is reflected."""
        from arize._generated.api_client.models.permission import Permission

        name = _unique("sdk-test-role")
        role = roles_client.create(
            name=name,
            permissions=[Permission.PROJECT_READ],
        )
        try:
            updated = roles_client.update(
                role_id=role.id,
                permissions=[
                    Permission.PROJECT_READ,
                    Permission.DATASET_CREATE,
                ],
            )
            assert Permission.DATASET_CREATE in updated.permissions
        finally:
            roles_client.delete(role=role.id)
