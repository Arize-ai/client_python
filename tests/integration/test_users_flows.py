"""Integration tests for UsersClient end-to-end flows against the real Arize API.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Run with:
    ARIZE_API_KEY=<key> \
        pytest tests/integration/test_users_flows.py -m integration -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from arize.organizations.types import PredefinedOrgRole
from arize.spaces.types import PredefinedSpaceRole
from arize.users.types import DeletionStatus, PredefinedUserRole

API_KEY = os.environ.get("ARIZE_API_KEY", "")
ORG_ID = os.environ.get("ARIZE_TEST_ORG_ID", "")
SPACE_ID = os.environ.get("ARIZE_TEST_SPACE_ID", "")

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
def users_client(arize_client) -> Any:
    return arize_client.users


_MEMBER_ROLE = PredefinedUserRole(name="member")


class TestUsersCRUD:
    """End-to-end CRUD flows for UsersClient."""

    def test_create_get_by_id(self, users_client) -> None:
        """Create a user, retrieve it by ID."""
        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = users_client.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )
        try:
            assert user.name == name
            assert user.id is not None

            fetched = users_client.get(user=user.id)
            assert fetched.id == user.id
            assert fetched.name == name
        finally:
            users_client.delete(user_id=user.id)

    def test_create_with_is_developer(self, users_client) -> None:
        """Create a user with is_developer=False and verify the field is set."""
        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = users_client.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
            is_developer=False,
        )
        try:
            assert user.is_developer is False
            fetched = users_client.get(user=user.id)
            assert fetched.is_developer is False
        finally:
            users_client.delete(user_id=user.id)

    def test_create_update(self, users_client) -> None:
        """Create a user, update name and is_developer."""
        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = users_client.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )
        updated_name = _unique("sdk-test-user-upd")
        try:
            updated = users_client.update(
                user_id=user.id,
                name=updated_name,
                is_developer=True,
            )
            assert updated.id == user.id
            assert updated.name == updated_name

            fetched = users_client.get(user=user.id)
            assert fetched.name == updated_name
        finally:
            users_client.delete(user_id=user.id)

    def test_create_appears_in_list(self, users_client) -> None:
        """Newly created user appears in list() results."""
        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = users_client.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )
        try:
            resp = users_client.list(limit=100)
            user_ids = [u.id for u in resp.users]
            assert user.id in user_ids
        finally:
            users_client.delete(user_id=user.id)

    def test_list_filter_by_email(self, users_client) -> None:
        """list(email=...) filters to users whose email contains the substring."""
        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = users_client.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )
        try:
            resp = users_client.list(email=email)
            assert any(u.id == user.id for u in resp.users)
        finally:
            users_client.delete(user_id=user.id)


class TestUsersGetByEmail:
    """End-to-end get-by-email flows for UsersClient."""

    def test_get_by_email(self, users_client) -> None:
        """Create a user, retrieve them by exact email, then delete."""
        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = users_client.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )
        try:
            found = users_client.get(user=email)
            assert found is not None
            assert found.id == user.id
            assert found.email.lower() == email.lower()
        finally:
            users_client.delete(user_id=user.id)

    def test_get_by_email_returns_none_for_unknown_email(
        self, users_client
    ) -> None:
        """get() returns None when no user matches the email."""
        result = users_client.get(user="nonexistent-xyz@sdk-test.arize.com")
        assert result is None


class TestUsersDelete:
    """End-to-end delete flows for UsersClient."""

    def test_create_delete(self, users_client) -> None:
        """Delete a user by ID; subsequent get raises a 404."""
        from arize._generated.api_client.exceptions import ApiException

        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = users_client.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )

        result = users_client.delete(user_id=user.id)

        assert result is None
        with pytest.raises(ApiException) as exc_info:
            users_client.get(user=user.id)
        assert exc_info.value.status == 404


class TestUsersResendInvitation:
    """End-to-end resend invitation flows for UsersClient."""

    def test_resend_invitation(self, users_client) -> None:
        """Create a user with email_link invite, resend invitation, then delete."""
        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = users_client.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="email_link",
        )
        try:
            users_client.resend_invitation(user_id=user.id)
        finally:
            users_client.delete(user_id=user.id)


class TestUsersResetPassword:
    """End-to-end reset password flows for UsersClient."""

    def test_reset_password(self, users_client) -> None:
        """Create an active user (invite_mode=none), trigger password reset, then delete."""
        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = users_client.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )
        try:
            result = users_client.reset_password(user_id=user.id)
            assert result is None
        finally:
            users_client.delete(user_id=user.id)


class TestUsersBulkDelete:
    """End-to-end bulk_delete flows for UsersClient."""

    def test_bulk_delete_by_ids(self, users_client) -> None:
        """Create two users, bulk-delete by ID, verify both deleted."""
        from arize._generated.api_client.exceptions import ApiException

        users = []
        for _ in range(2):
            name = _unique("sdk-test-user")
            email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
            users.append(
                users_client.create(
                    name=name,
                    email=email,
                    role=_MEMBER_ROLE,
                    invite_mode="none",
                )
            )

        results = users_client.bulk_delete(user_ids=[u.id for u in users])

        assert len(results) == 2
        assert all(r.status == DeletionStatus.DELETED for r in results)

        for u in users:
            with pytest.raises(ApiException) as exc_info:
                users_client.get(user_id=u.id)
            assert exc_info.value.status == 404

    def test_bulk_delete_by_email(self, users_client) -> None:
        """Create a user, bulk-delete by email, verify deleted."""
        from arize._generated.api_client.exceptions import ApiException

        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = users_client.create(
            name=_unique("sdk-test-user"),
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )

        results = users_client.bulk_delete(emails=[email])

        assert len(results) == 1
        assert results[0].status == DeletionStatus.DELETED
        assert results[0].id == user.id

        with pytest.raises(ApiException) as exc_info:
            users_client.get(user_id=user.id)
        assert exc_info.value.status == 404

    def test_bulk_delete_email_not_found(self, users_client) -> None:
        """Deleting a nonexistent email produces a not_found entry."""
        results = users_client.bulk_delete(
            emails=["nonexistent-user@sdk-test.arize.com"]
        )

        assert len(results) == 1
        assert results[0].status == DeletionStatus.NOT_FOUND
        assert results[0].error is not None

    def test_bulk_delete_mixed(self, users_client) -> None:
        """Mix of valid IDs and nonexistent emails in one call."""
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = users_client.create(
            name=_unique("sdk-test-user"),
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )

        results = users_client.bulk_delete(
            user_ids=[user.id],
            emails=["nonexistent-user@sdk-test.arize.com"],
        )

        statuses = {r.status for r in results}
        assert DeletionStatus.DELETED in statuses
        assert DeletionStatus.NOT_FOUND in statuses


@pytest.mark.skipif(
    not ORG_ID,
    reason="ARIZE_TEST_ORG_ID must be set",
)
class TestOrganizationAddUser:
    """End-to-end add/remove user flows for organizations."""

    def test_add_user_to_organization(self, arize_client) -> None:
        """Create a user, add them to an existing organization, then clean up."""
        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = arize_client.users.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )
        try:
            membership = arize_client.organizations.add_user(
                organization=ORG_ID,
                user_id=user.id,
                role=PredefinedOrgRole(name="member"),
            )
            assert membership.user_id == user.id
            assert membership.organization_id == ORG_ID

            result = arize_client.organizations.remove_user(
                organization=ORG_ID,
                user_id=user.id,
            )
            assert result is None
        finally:
            arize_client.users.delete(user_id=user.id)

    def test_add_user_upserts_role(self, arize_client) -> None:
        """Adding a user twice updates their role (upsert semantics)."""
        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = arize_client.users.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )
        try:
            arize_client.organizations.add_user(
                organization=ORG_ID,
                user_id=user.id,
                role=PredefinedOrgRole(name="member"),
            )
            updated = arize_client.organizations.add_user(
                organization=ORG_ID,
                user_id=user.id,
                role=PredefinedOrgRole(name="read-only"),
            )
            assert updated.user_id == user.id
        finally:
            with __import__("contextlib").suppress(Exception):
                arize_client.organizations.remove_user(
                    organization=ORG_ID, user_id=user.id
                )
            arize_client.users.delete(user_id=user.id)


@pytest.mark.skipif(
    not ORG_ID or not SPACE_ID,
    reason="ARIZE_TEST_ORG_ID and ARIZE_TEST_SPACE_ID must be set",
)
class TestSpaceAddUser:
    """End-to-end add/remove user flows for spaces."""

    def test_add_user_to_space(self, arize_client) -> None:
        """Create a user, add them to org then space, verify membership, then clean up."""
        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = arize_client.users.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )
        try:
            # User must be in the org before being added to the space
            arize_client.organizations.add_user(
                organization=ORG_ID,
                user_id=user.id,
                role=PredefinedOrgRole(name="member"),
            )

            membership = arize_client.spaces.add_user(
                space=SPACE_ID,
                user_id=user.id,
                role=PredefinedSpaceRole(name="member"),
            )
            assert membership.user_id == user.id
            assert membership.space_id == SPACE_ID

            result = arize_client.spaces.remove_user(
                space=SPACE_ID,
                user_id=user.id,
            )
            assert result is None
        finally:
            with __import__("contextlib").suppress(Exception):
                arize_client.organizations.remove_user(
                    organization=ORG_ID, user_id=user.id
                )
            arize_client.users.delete(user_id=user.id)

    def test_add_user_to_space_upserts_role(self, arize_client) -> None:
        """Adding a user to a space twice updates their role (upsert semantics)."""
        name = _unique("sdk-test-user")
        email = f"{uuid.uuid4().hex[:8]}@sdk-test.arize.com"
        user = arize_client.users.create(
            name=name,
            email=email,
            role=_MEMBER_ROLE,
            invite_mode="none",
        )
        try:
            arize_client.organizations.add_user(
                organization=ORG_ID,
                user_id=user.id,
                role=PredefinedOrgRole(name="member"),
            )
            arize_client.spaces.add_user(
                space=SPACE_ID,
                user_id=user.id,
                role=PredefinedSpaceRole(name="member"),
            )
            updated = arize_client.spaces.add_user(
                space=SPACE_ID,
                user_id=user.id,
                role=PredefinedSpaceRole(name="read-only"),
            )
            assert updated.user_id == user.id
        finally:
            with __import__("contextlib").suppress(Exception):
                arize_client.spaces.remove_user(space=SPACE_ID, user_id=user.id)
            with __import__("contextlib").suppress(Exception):
                arize_client.organizations.remove_user(
                    organization=ORG_ID, user_id=user.id
                )
            arize_client.users.delete(user_id=user.id)
