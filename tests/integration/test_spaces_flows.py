"""Integration tests for SpacesClient end-to-end flows against the real Arize API.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_ORG_ID=<org-id> \
        pytest tests/integration/test_spaces_flows.py -m integration -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from arize.utils.resolve import is_resource_id

API_KEY = os.environ.get("ARIZE_API_KEY", "")
ORG_ID = os.environ.get("ARIZE_TEST_ORG_ID", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY or not ORG_ID,
        reason="ARIZE_API_KEY and ARIZE_TEST_ORG_ID must be set",
    ),
]


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def spaces_client(arize_client: Any) -> Any:
    return arize_client.spaces


class TestSpacesCRUD:
    """End-to-end CRUD flows for SpacesClient."""

    def test_create_get_by_id(self, spaces_client: Any) -> None:
        """Create a space, retrieve it by ID."""
        name = _unique("sdk-test-space")
        space = spaces_client.create(name=name, organization_id=ORG_ID)
        try:
            assert space.name == name
            assert is_resource_id(space.id)

            fetched = spaces_client.get(space=space.id)
            assert fetched.id == space.id
            assert fetched.name == name
        finally:
            spaces_client.delete(space=space.id)

    def test_create_get_by_name(self, spaces_client: Any) -> None:
        """Create a space, retrieve it by name."""
        name = _unique("sdk-test-space")
        space = spaces_client.create(name=name, organization_id=ORG_ID)
        try:
            fetched = spaces_client.get(space=name)
            assert fetched.id == space.id
            assert fetched.name == name
        finally:
            spaces_client.delete(space=space.id)

    def test_create_appears_in_list(self, spaces_client: Any) -> None:
        """Newly created space appears in list() results."""
        name = _unique("sdk-test-space")
        space = spaces_client.create(name=name, organization_id=ORG_ID)
        try:
            resp = spaces_client.list(limit=100)
            space_ids = [s.id for s in resp.spaces]
            assert space.id in space_ids
        finally:
            spaces_client.delete(space=space.id)

    def test_create_update_name(self, spaces_client: Any) -> None:
        """Create a space then update its name."""
        original_name = _unique("sdk-test-space")
        space = spaces_client.create(
            name=original_name,
            organization_id=ORG_ID,
            description="original description",
        )
        updated_name = _unique("sdk-test-space-upd")
        try:
            updated = spaces_client.update(space=space.id, name=updated_name)
            assert updated.id == space.id
            assert updated.name == updated_name

            fetched = spaces_client.get(space=space.id)
            assert fetched.name == updated_name
        finally:
            spaces_client.delete(space=space.id)


class TestSpacesPrivateVisibility:
    """End-to-end flows for is_private on SpacesClient."""

    def test_create_public_space_has_is_private_false(
        self, spaces_client: Any
    ) -> None:
        """Spaces created without is_private default to public (is_private=False)."""
        name = _unique("sdk-test-space-pub")
        space = spaces_client.create(name=name, organization_id=ORG_ID)
        try:
            assert space.is_private is False
            fetched = spaces_client.get(space=space.id)
            assert fetched.is_private is False
        finally:
            spaces_client.delete(space=space.id)

    def test_create_private_space(self, spaces_client: Any) -> None:
        """Create a space with is_private=True; verify the flag is persisted."""
        name = _unique("sdk-test-space-priv")
        space = spaces_client.create(
            name=name, organization_id=ORG_ID, is_private=True
        )
        try:
            assert space.is_private is True
            fetched = spaces_client.get(space=space.id)
            assert fetched.is_private is True
        finally:
            spaces_client.delete(space=space.id)

    def test_update_space_toggle_to_private(self, spaces_client: Any) -> None:
        """Create a public space then toggle it to private via update."""
        name = _unique("sdk-test-space-toggle")
        space = spaces_client.create(name=name, organization_id=ORG_ID)
        try:
            assert space.is_private is False

            updated = spaces_client.update(space=space.id, is_private=True)
            assert updated.is_private is True

            fetched = spaces_client.get(space=space.id)
            assert fetched.is_private is True
        finally:
            spaces_client.delete(space=space.id)

    def test_update_space_toggle_back_to_public(
        self, spaces_client: Any
    ) -> None:
        """Create a private space then toggle it back to public via update."""
        name = _unique("sdk-test-space-unpub")
        space = spaces_client.create(
            name=name, organization_id=ORG_ID, is_private=True
        )
        try:
            assert space.is_private is True

            updated = spaces_client.update(space=space.id, is_private=False)
            assert updated.is_private is False

            fetched = spaces_client.get(space=space.id)
            assert fetched.is_private is False
        finally:
            spaces_client.delete(space=space.id)

    def test_update_name_preserves_is_private(self, spaces_client: Any) -> None:
        """Updating only the name does not change is_private."""
        name = _unique("sdk-test-space-pname")
        space = spaces_client.create(
            name=name, organization_id=ORG_ID, is_private=True
        )
        new_name = _unique("sdk-test-space-pname-upd")
        try:
            updated = spaces_client.update(space=space.id, name=new_name)
            assert updated.name == new_name
            assert updated.is_private is True
        finally:
            spaces_client.delete(space=space.id)
