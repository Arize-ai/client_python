"""Unit tests for src/arize/users/client.py."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from arize.users.client import UsersClient
from arize.users.types import (
    BulkUserDeletionResult,
    CustomUserRole,
    DeletionStatus,
    PredefinedUserRole,
    User,
    UserRole,
    UsersList200Response,
)


@pytest.fixture(autouse=True)
def _stub_from_generated() -> Generator[None, None, None]:
    """Stub model_validate on all domain types so tests that don't explicitly
    test conversion don't fail when the client calls it on a Mock API response.
    """
    with (
        patch.object(User, "model_validate", return_value=Mock()),
        patch.object(
            UsersList200Response, "model_validate", return_value=Mock()
        ),
    ):
        yield


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock UsersApi instance."""
    return Mock()


@pytest.fixture
def users_client(mock_sdk_config: Mock, mock_api: Mock) -> UsersClient:
    """Provide a UsersClient with mocked internals."""
    with patch("arize._generated.api_client.UsersApi", return_value=mock_api):
        return UsersClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestUsersClientInit:
    """Tests for UsersClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.UsersApi",
            return_value=mock_api,
        ):
            client = UsersClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_users_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to UsersApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.UsersApi"
        ) as mock_users_api_cls:
            UsersClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_users_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestUsersClientList:
    """Tests for UsersClient.list()."""

    def test_list_calls_api_with_all_params(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """list() should pass email, status, limit, and cursor to users_list."""
        users_client.list(
            email="jane@example.com",
            status=["active", "invited"],
            limit=25,
            cursor="cursor-abc",
        )

        mock_api.users_list.assert_called_once_with(
            limit=25,
            cursor="cursor-abc",
            email="jane@example.com",
            status=["active", "invited"],
        )

    def test_list_defaults(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """list() should default email/status/cursor to None and limit to 50."""
        users_client.list()

        mock_api.users_list.assert_called_once_with(
            limit=50,
            cursor=None,
            email=None,
            status=None,
        )

    def test_list_returns_domain_response(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """list() should convert the raw API response to a domain UsersList200Response."""
        raw = Mock()
        mock_api.users_list.return_value = raw
        domain = Mock()
        with patch.object(
            UsersList200Response, "model_validate", return_value=domain
        ) as mock_conv:
            result = users_client.list()
        mock_conv.assert_called_once_with(raw, from_attributes=True)
        assert result is domain

    def test_list_emits_alpha_prerelease_warning(
        self,
        users_client: UsersClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        users_client.list()

        assert any(
            "ALPHA" in record.message and "users.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestUsersClientGet:
    """Tests for UsersClient.get()."""

    def test_get_by_id_calls_api(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """get() with an ID should pass it directly to users_get."""
        users_client.get(user="user-12345")

        mock_api.users_get.assert_called_once_with(user_id="user-12345")

    def test_get_by_id_returns_domain_user(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """get() by ID should convert the raw API response to a domain User."""
        raw = Mock()
        mock_api.users_get.return_value = raw
        domain = Mock()
        with patch.object(
            User, "model_validate", return_value=domain
        ) as mock_conv:
            result = users_client.get(user="user-12345")
        mock_conv.assert_called_once_with(raw, from_attributes=True)
        assert result is domain

    def test_get_by_email_returns_matching_user(
        self, users_client: UsersClient
    ) -> None:
        """get() with an email returns the user whose email matches exactly."""
        target = Mock()
        target.email = "alice@example.com"
        other = Mock()
        other.email = "bob@example.com"
        with patch.object(
            users_client,
            "list",
            return_value=Mock(users=[other, target]),
        ):
            result = users_client.get(user="alice@example.com")
        assert result is target

    def test_get_by_email_returns_none_when_no_match(
        self, users_client: UsersClient
    ) -> None:
        """get() with an email returns None when no exact match is found."""
        user = Mock()
        user.email = "bob@example.com"
        with patch.object(
            users_client,
            "list",
            return_value=Mock(users=[user]),
        ):
            result = users_client.get(user="alice@example.com")
        assert result is None

    def test_get_by_email_case_insensitive(
        self, users_client: UsersClient
    ) -> None:
        """Email comparison in get() should be case-insensitive."""
        target = Mock()
        target.email = "Alice@Example.COM"
        with patch.object(
            users_client,
            "list",
            return_value=Mock(users=[target]),
        ):
            result = users_client.get(user="alice@example.com")
        assert result is target

    def test_get_by_email_no_false_positive_on_substring(
        self, users_client: UsersClient
    ) -> None:
        """get() by email must not return a user whose email is only a substring match."""
        user = Mock()
        user.email = "xfoo@bar.com"
        with patch.object(
            users_client,
            "list",
            return_value=Mock(users=[user]),
        ):
            result = users_client.get(user="foo@bar.com")
        assert result is None

    def test_get_by_email_does_not_call_users_get_api(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """get() with an email should resolve via list(), not users_get."""
        target = Mock()
        target.email = "alice@example.com"
        with patch.object(
            users_client,
            "list",
            return_value=Mock(users=[target]),
        ):
            users_client.get(user="alice@example.com")
        mock_api.users_get.assert_not_called()

    def test_get_emits_alpha_prerelease_warning(
        self,
        users_client: UsersClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        users_client.get(user="user-12345")

        assert any(
            "ALPHA" in record.message and "users.get" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestUsersClientCreate:
    """Tests for UsersClient.create()."""

    def test_create_predefined_role_builds_request_and_calls_api(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """create() should wrap a PredefinedUserRole in PredefinedUserRoleAssignment and UserRoleAssignment."""
        role = PredefinedUserRole(name=UserRole.MEMBER)
        with (
            patch(
                "arize._generated.api_client.CreateUserRequest"
            ) as mock_request_cls,
            patch(
                "arize._generated.api_client.UserRoleAssignment"
            ) as mock_role_cls,
            patch(
                "arize._generated.api_client.PredefinedUserRoleAssignment"
            ) as mock_pred_cls,
        ):
            mock_body = Mock()
            mock_request_cls.return_value = mock_body
            mock_role = Mock()
            mock_role_cls.return_value = mock_role
            mock_gen_role = Mock()
            mock_pred_cls.return_value = mock_gen_role

            users_client.create(
                name="Jane Smith",
                email="jane@example.com",
                role=role,
                invite_mode="email_link",
            )

        mock_pred_cls.assert_called_once()
        mock_role_cls.assert_called_once_with(mock_gen_role)
        mock_request_cls.assert_called_once_with(
            name="Jane Smith",
            email="jane@example.com",
            role=mock_role,
            invite_mode="email_link",
        )
        mock_api.users_create.assert_called_once_with(
            create_user_request=mock_body
        )

    def test_create_custom_role_builds_request_and_calls_api(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """create() should wrap a CustomUserRole in CustomUserRoleAssignment and UserRoleAssignment."""
        role = CustomUserRole(id="role-abc-123")
        with (
            patch(
                "arize._generated.api_client.CreateUserRequest"
            ) as mock_request_cls,
            patch(
                "arize._generated.api_client.UserRoleAssignment"
            ) as mock_role_cls,
            patch(
                "arize._generated.api_client.CustomUserRoleAssignment"
            ) as mock_custom_cls,
        ):
            mock_body = Mock()
            mock_request_cls.return_value = mock_body
            mock_role = Mock()
            mock_role_cls.return_value = mock_role
            mock_gen_role = Mock()
            mock_custom_cls.return_value = mock_gen_role

            users_client.create(
                name="Jane Smith",
                email="jane@example.com",
                role=role,
                invite_mode="email_link",
            )

        mock_custom_cls.assert_called_once()
        mock_role_cls.assert_called_once_with(mock_gen_role)
        mock_request_cls.assert_called_once_with(
            name="Jane Smith",
            email="jane@example.com",
            role=mock_role,
            invite_mode="email_link",
        )
        mock_api.users_create.assert_called_once_with(
            create_user_request=mock_body
        )

    def test_create_forwards_is_developer_when_provided(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """create() should include is_developer in CreateUserRequest when provided."""
        role = PredefinedUserRole(name=UserRole.MEMBER)
        with (
            patch(
                "arize._generated.api_client.CreateUserRequest"
            ) as mock_request_cls,
            patch(
                "arize._generated.api_client.UserRoleAssignment"
            ) as mock_role_cls,
        ):
            mock_role = Mock()
            mock_role_cls.return_value = mock_role

            users_client.create(
                name="Jane Smith",
                email="jane@example.com",
                role=role,
                invite_mode="email_link",
                is_developer=True,
            )

        mock_request_cls.assert_called_once_with(
            name="Jane Smith",
            email="jane@example.com",
            role=mock_role,
            invite_mode="email_link",
            is_developer=True,
        )

    def test_create_omits_is_developer_when_none(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """create() should omit is_developer from CreateUserRequest when not provided."""
        role = PredefinedUserRole(name=UserRole.MEMBER)
        with (
            patch(
                "arize._generated.api_client.CreateUserRequest"
            ) as mock_request_cls,
            patch(
                "arize._generated.api_client.UserRoleAssignment"
            ) as mock_role_cls,
        ):
            mock_role = Mock()
            mock_role_cls.return_value = mock_role

            users_client.create(
                name="Jane Smith",
                email="jane@example.com",
                role=role,
                invite_mode="email_link",
            )

        mock_request_cls.assert_called_once_with(
            name="Jane Smith",
            email="jane@example.com",
            role=mock_role,
            invite_mode="email_link",
        )

    def test_create_returns_domain_user(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """create() should convert the raw API response to a domain User."""
        raw = Mock()
        mock_api.users_create.return_value = raw
        domain = Mock()

        with (
            patch("arize._generated.api_client.CreateUserRequest"),
            patch("arize._generated.api_client.UserRoleAssignment"),
            patch.object(
                User, "model_validate", return_value=domain
            ) as mock_conv,
        ):
            result = users_client.create(
                name="Jane Smith",
                email="jane@example.com",
                role=PredefinedUserRole(name=UserRole.MEMBER),
                invite_mode="none",
            )

        mock_conv.assert_called_once_with(raw, from_attributes=True)
        assert result is domain

    def test_create_emits_alpha_prerelease_warning(
        self,
        users_client: UsersClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch("arize._generated.api_client.CreateUserRequest"),
            patch("arize._generated.api_client.UserRoleAssignment"),
        ):
            users_client.create(
                name="Jane Smith",
                email="jane@example.com",
                role=PredefinedUserRole(name=UserRole.MEMBER),
                invite_mode="email_link",
            )

        assert any(
            "ALPHA" in record.message and "users.create" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestUsersClientUpdate:
    """Tests for UsersClient.update()."""

    def test_update_raises_when_no_fields_provided(
        self, users_client: UsersClient
    ) -> None:
        """update() should raise if neither name nor is_developer is provided."""
        with pytest.raises(
            ValueError,
            match="At least one of 'name' or 'is_developer' must be provided",
        ):
            users_client.update(user_id="user-12345")

    def test_update_builds_request_and_calls_api(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """update() should build UserUpdate and pass it to users_update."""
        with patch(
            "arize._generated.api_client.UserUpdate"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            users_client.update(
                user_id="user-12345",
                name="Updated Name",
                is_developer=True,
            )

        mock_request_cls.assert_called_once_with(
            name="Updated Name",
            is_developer=True,
        )
        mock_api.users_update.assert_called_once_with(
            user_id="user-12345",
            user_update=mock_body,
        )

    def test_update_returns_domain_user(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """update() should convert the raw API response to a domain User."""
        raw = Mock()
        mock_api.users_update.return_value = raw
        domain = Mock()

        with (
            patch("arize._generated.api_client.UserUpdate"),
            patch.object(
                User, "model_validate", return_value=domain
            ) as mock_conv,
        ):
            result = users_client.update(
                user_id="user-12345",
                name="Updated Name",
            )

        mock_conv.assert_called_once_with(raw, from_attributes=True)
        assert result is domain

    def test_update_emits_alpha_prerelease_warning(
        self,
        users_client: UsersClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch("arize._generated.api_client.UserUpdate"):
            users_client.update(user_id="user-12345", name="Updated Name")

        assert any(
            "ALPHA" in record.message and "users.update" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestUsersClientDelete:
    """Tests for UsersClient.delete()."""

    def test_delete_calls_api_with_user_id(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """delete() should pass user_id to users_delete."""
        users_client.delete(user_id="user-12345")

        mock_api.users_delete.assert_called_once_with(user_id="user-12345")

    def test_delete_returns_none(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """delete() should return None on success (204 response)."""
        mock_api.users_delete.return_value = None

        result = users_client.delete(user_id="user-12345")

        assert result is None

    def test_delete_emits_alpha_prerelease_warning(
        self,
        users_client: UsersClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        users_client.delete(user_id="user-12345")

        assert any(
            "ALPHA" in record.message and "users.delete" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestUsersClientResendInvitation:
    """Tests for UsersClient.resend_invitation()."""

    def test_resend_invitation_calls_api_with_user_id(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """resend_invitation() should pass user_id to users_resend_invitation."""
        users_client.resend_invitation(user_id="user-12345")

        mock_api.users_resend_invitation.assert_called_once_with(
            user_id="user-12345",
        )

    def test_resend_invitation_returns_none(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """resend_invitation() should return None on success (202 response)."""
        mock_api.users_resend_invitation.return_value = None

        result = users_client.resend_invitation(user_id="user-12345")

        assert result is None

    def test_resend_invitation_emits_alpha_prerelease_warning(
        self,
        users_client: UsersClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        users_client.resend_invitation(user_id="user-12345")

        assert any(
            "ALPHA" in record.message
            and "users.resend_invitation" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestUsersClientResetPassword:
    """Tests for UsersClient.reset_password()."""

    def test_reset_password_calls_api_with_user_id(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """reset_password() should pass user_id to users_password_reset."""
        users_client.reset_password(user_id="user-12345")

        mock_api.users_password_reset.assert_called_once_with(
            user_id="user-12345",
        )

    def test_reset_password_returns_none(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """reset_password() should return None on success (204 response)."""
        mock_api.users_password_reset.return_value = None

        result = users_client.reset_password(user_id="user-12345")

        assert result is None

    def test_reset_password_emits_alpha_prerelease_warning(
        self,
        users_client: UsersClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        users_client.reset_password(user_id="user-12345")

        assert any(
            "ALPHA" in record.message
            and "users.reset_password" in record.message
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# Helpers for bulk_delete tests
# ---------------------------------------------------------------------------


def _make_users_list_response(
    users: list[tuple[str, str]],
) -> Mock:
    """Return a mock ``users_list`` API response.

    Each tuple is ``(user_id, email)``.  The response has a
    ``pagination`` with no ``next_cursor`` so the resolver stops
    after one page.
    """
    resp = Mock()
    resp.users = []
    for uid, em in users:
        u = Mock()
        u.id = uid
        u.email = em
        resp.users.append(u)
    resp.pagination = Mock(spec=["next_cursor"])
    resp.pagination.next_cursor = None
    return resp


@pytest.mark.unit
class TestUsersClientBulkDelete:
    """Tests for UsersClient.bulk_delete()."""

    def test_bulk_delete_by_ids(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """Providing user_ids should delete each and return 'deleted'."""
        mock_api.users_delete.return_value = None

        results = users_client.bulk_delete(user_ids=["u1", "u2"])

        assert len(results) == 2
        assert all(r.status == DeletionStatus.DELETED for r in results)
        assert [r.id for r in results] == ["u1", "u2"]
        assert mock_api.users_delete.call_count == 2

    def test_bulk_delete_by_emails(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """Emails should be resolved then deleted."""
        mock_api.users_list.return_value = _make_users_list_response(
            [("resolved-1", "alice@example.com")]
        )
        mock_api.users_delete.return_value = None

        results = users_client.bulk_delete(emails=["alice@example.com"])

        assert len(results) == 1
        assert results[0] == BulkUserDeletionResult(
            id="resolved-1", status=DeletionStatus.DELETED
        )

    def test_bulk_delete_email_not_found(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """Unresolvable emails should produce a 'not_found' entry."""
        mock_api.users_list.return_value = _make_users_list_response([])

        results = users_client.bulk_delete(emails=["ghost@example.com"])

        assert len(results) == 1
        assert results[0].id == "ghost@example.com"
        assert results[0].status == DeletionStatus.NOT_FOUND
        assert results[0].error is not None

    def test_bulk_delete_partial_failure(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """One success and one failure in a single results list."""

        def _side_effect(*, user_id: str) -> None:
            if user_id == "bad":
                raise RuntimeError("API exploded")

        mock_api.users_delete.side_effect = _side_effect

        results = users_client.bulk_delete(user_ids=["good", "bad"])

        assert len(results) == 2
        assert results[0] == BulkUserDeletionResult(
            id="good", status=DeletionStatus.DELETED
        )
        assert results[1].id == "bad"
        assert results[1].status == DeletionStatus.FAILED
        assert "API exploded" in (results[1].error or "")

    def test_bulk_delete_raises_if_no_args(
        self, users_client: UsersClient
    ) -> None:
        """Calling with neither user_ids nor emails raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            users_client.bulk_delete()

    def test_bulk_delete_email_case_insensitive(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """Email resolution should be case-insensitive."""
        mock_api.users_list.return_value = _make_users_list_response(
            [("uid-1", "Alice@Example.COM")]
        )
        mock_api.users_delete.return_value = None

        results = users_client.bulk_delete(emails=["alice@example.com"])

        assert len(results) == 1
        assert results[0].id == "uid-1"
        assert results[0].status == "deleted"

    def test_bulk_delete_logs_not_found_emails(
        self,
        users_client: UsersClient,
        mock_api: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unresolvable emails should log a warning."""
        mock_api.users_list.return_value = _make_users_list_response([])
        caplog.set_level(logging.WARNING)

        users_client.bulk_delete(emails=["ghost@example.com"])

        assert any(
            "ghost@example.com" in record.message for record in caplog.records
        )

    def test_bulk_delete_emits_alpha_prerelease_warning(
        self,
        users_client: UsersClient,
        mock_api: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        mock_api.users_delete.return_value = None
        users_client.bulk_delete(user_ids=["u1"])

        assert any(
            "ALPHA" in record.message and "users.bulk_delete" in record.message
            for record in caplog.records
        )
