"""Integration tests for OrganizationsClient end-to-end flows against the real Arize API.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Run with:
    ARIZE_API_KEY=<key> \
        pytest tests/integration/test_organizations_flows.py -m integration -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from arize.utils.resolve import is_resource_id

API_KEY = os.environ.get("ARIZE_API_KEY", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY,
        reason="ARIZE_API_KEY must be set",
    ),
]


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def organizations_client(arize_client) -> Any:
    return arize_client.organizations


class TestOrganizationsCRUD:
    """End-to-end CRUD flows for OrganizationsClient."""

    def test_create_get_by_id(self, organizations_client) -> None:
        """Create an organization, retrieve it by ID."""
        name = _unique("sdk-test-org")
        org = organizations_client.create(
            name=name,
            description="Created by SDK integration test",
        )
        try:
            assert org.name == name
            assert is_resource_id(org.id)
            assert org.description == "Created by SDK integration test"

            fetched = organizations_client.get(organization=org.id)
            assert fetched.id == org.id
            assert fetched.name == name
        finally:
            organizations_client.delete(organization=org.id)

    def test_create_get_by_name(self, organizations_client) -> None:
        """Create an organization, retrieve it by name."""
        name = _unique("sdk-test-org")
        org = organizations_client.create(name=name)
        try:
            fetched = organizations_client.get(organization=name)
            assert fetched.id == org.id
            assert fetched.name == name
        finally:
            organizations_client.delete(organization=org.id)

    def test_create_update(self, organizations_client) -> None:
        """Create an organization, update its name and description."""
        original_name = _unique("sdk-test-org")
        org = organizations_client.create(
            name=original_name,
            description="Original description",
        )
        updated_name = _unique("sdk-test-org-upd")
        try:
            updated = organizations_client.update(
                organization=org.id,
                name=updated_name,
                description="Updated description",
            )
            assert updated.id == org.id
            assert updated.name == updated_name
            assert updated.description == "Updated description"

            fetched = organizations_client.get(organization=org.id)
            assert fetched.name == updated_name
        finally:
            organizations_client.delete(organization=org.id)

    def test_create_appears_in_list(self, organizations_client) -> None:
        """Newly created organization appears in list() results."""
        name = _unique("sdk-test-org")
        org = organizations_client.create(name=name)
        try:
            resp = organizations_client.list(limit=100)
            org_ids = [o.id for o in resp.organizations]
            assert org.id in org_ids
        finally:
            organizations_client.delete(organization=org.id)

    def test_list_filter_by_name(self, organizations_client) -> None:
        """list(name=...) filters to organizations whose names contain the substring."""
        name = _unique("sdk-test-org")
        org = organizations_client.create(name=name)
        try:
            resp = organizations_client.list(name=name)
            assert any(o.id == org.id for o in resp.organizations)
        finally:
            organizations_client.delete(organization=org.id)


class TestOrganizationsClientDelete:
    """End-to-end delete flows for OrganizationsClient."""

    def test_create_delete_by_id(self, organizations_client) -> None:
        """Delete an organization by ID; subsequent get raises a 404."""
        from arize._generated.api_client.exceptions import ApiException

        name = _unique("sdk-test-org")
        org = organizations_client.create(name=name)

        result = organizations_client.delete(organization=org.id)

        assert result is None
        with pytest.raises(ApiException) as exc_info:
            organizations_client.get(organization=org.id)
        assert exc_info.value.status == 404

    def test_create_delete_by_name(self, organizations_client) -> None:
        """Delete an organization by name; subsequent get raises a 404."""
        from arize._generated.api_client.exceptions import ApiException

        name = _unique("sdk-test-org")
        org = organizations_client.create(name=name)

        result = organizations_client.delete(organization=name)

        assert result is None
        with pytest.raises(ApiException) as exc_info:
            organizations_client.get(organization=org.id)
        assert exc_info.value.status == 404
