"""Unit tests for src/arize/users/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.users.client import UsersClient
from arize.users.types import CustomUserRole, PredefinedUserRole, UserRole


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

    def test_list_returns_api_response(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from users_list."""
        expected = Mock()
        mock_api.users_list.return_value = expected

        result = users_client.list()

        assert result is expected

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

    def test_get_calls_api_with_user_id(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """get() should pass user_id to users_get."""
        users_client.get(user_id="user-12345")

        mock_api.users_get.assert_called_once_with(user_id="user-12345")

    def test_get_returns_api_response(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """get() should propagate the return value from users_get."""
        expected = Mock()
        mock_api.users_get.return_value = expected

        result = users_client.get(user_id="user-12345")

        assert result is expected

    def test_get_emits_alpha_prerelease_warning(
        self,
        users_client: UsersClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        users_client.get(user_id="user-12345")

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
        """create() should convert PredefinedUserRole via _to_generated() and UserRoleAssignment."""
        role = PredefinedUserRole(name=UserRole.MEMBER)
        with (
            patch(
                "arize._generated.api_client.CreateUserRequest"
            ) as mock_request_cls,
            patch(
                "arize._generated.api_client.UserRoleAssignment"
            ) as mock_role_cls,
        ):
            mock_body = Mock()
            mock_request_cls.return_value = mock_body
            mock_role = Mock()
            mock_role_cls.return_value = mock_role

            users_client.create(
                name="Jane Smith",
                email="jane@example.com",
                role=role,
                invite_mode="email_link",
            )

        mock_role_cls.assert_called_once_with(role._to_generated())
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
        """create() should convert CustomUserRole via _to_generated() and UserRoleAssignment."""
        role = CustomUserRole(id="role-abc-123")
        with (
            patch(
                "arize._generated.api_client.CreateUserRequest"
            ) as mock_request_cls,
            patch(
                "arize._generated.api_client.UserRoleAssignment"
            ) as mock_role_cls,
        ):
            mock_body = Mock()
            mock_request_cls.return_value = mock_body
            mock_role = Mock()
            mock_role_cls.return_value = mock_role

            users_client.create(
                name="Jane Smith",
                email="jane@example.com",
                role=role,
                invite_mode="email_link",
            )

        mock_role_cls.assert_called_once_with(role._to_generated())
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

    def test_create_returns_api_response(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from users_create."""
        expected = Mock()
        mock_api.users_create.return_value = expected

        with (
            patch("arize._generated.api_client.CreateUserRequest"),
            patch("arize._generated.api_client.UserRoleAssignment"),
        ):
            result = users_client.create(
                name="Jane Smith",
                email="jane@example.com",
                role=PredefinedUserRole(name=UserRole.MEMBER),
                invite_mode="none",
            )

        assert result is expected

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

    def test_update_returns_api_response(
        self, users_client: UsersClient, mock_api: Mock
    ) -> None:
        """update() should propagate the return value from users_update."""
        expected = Mock()
        mock_api.users_update.return_value = expected

        with patch("arize._generated.api_client.UserUpdate"):
            result = users_client.update(
                user_id="user-12345",
                name="Updated Name",
            )

        assert result is expected

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
        """resend_invitation() should pass user_id and Content-Type header to users_resend_invitation."""
        users_client.resend_invitation(user_id="user-12345")

        mock_api.users_resend_invitation.assert_called_once_with(
            user_id="user-12345",
            _headers={"Content-Type": "application/json"},
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
        """reset_password() should pass user_id and Content-Type header to users_password_reset."""
        users_client.reset_password(user_id="user-12345")

        mock_api.users_password_reset.assert_called_once_with(
            user_id="user-12345",
            _headers={"Content-Type": "application/json"},
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
