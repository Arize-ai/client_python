"""Unit tests for src/arize/api_keys/client.py."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import Mock, create_autospec, patch

import pytest

from arize._generated.api_client import APIKeysApi
from arize.api_keys.client import ApiKeysClient


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock ApiKeysApi instance."""
    return create_autospec(APIKeysApi, instance=True)


@pytest.fixture
def api_keys_client(mock_sdk_config: Mock, mock_api: Mock) -> ApiKeysClient:
    """Provide an ApiKeysClient with mocked internals."""
    with patch("arize._generated.api_client.APIKeysApi", return_value=mock_api):
        return ApiKeysClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestApiKeysClientInit:
    """Tests for ApiKeysClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.APIKeysApi", return_value=mock_api
        ):
            client = ApiKeysClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_api_keys_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to APIKeysApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.APIKeysApi"
        ) as mock_api_keys_api_cls:
            ApiKeysClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_api_keys_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestApiKeysClientList:
    """Tests for ApiKeysClient.list()."""

    # Base64 ID that decodes to "Space:905:abc" — passes is_resource_id()
    _SPACE_ID = "U3BhY2U6OTA1MDoxSmtS"

    def test_list_calls_api_with_all_params(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """list() should forward all parameters to api_keys_list."""
        api_keys_client.list(
            key_type="SERVICE",
            status="ACTIVE",
            space=self._SPACE_ID,
            user_id="VXNlcjoxMjM0NQ==",
            limit=25,
            cursor="cursor-abc",
        )

        mock_api.list_api_keys.assert_called_once_with(
            key_type="SERVICE",
            status="ACTIVE",
            space_id=self._SPACE_ID,
            user_id="VXNlcjoxMjM0NQ==",
            limit=25,
            cursor="cursor-abc",
        )

    def test_list_defaults(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """list() should default all optional params to None and limit to 50."""
        api_keys_client.list()

        mock_api.list_api_keys.assert_called_once_with(
            key_type=None,
            status=None,
            space_id=None,
            user_id=None,
            limit=50,
            cursor=None,
        )

    def test_list_with_space_name_resolves_to_id(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """list() should resolve a space name to an ID via _find_space_id."""
        with patch(
            "arize.api_keys.client._find_space_id",
            return_value="resolved-space-id",
        ) as mock_resolve:
            api_keys_client.list(space="my-space")

        mock_resolve.assert_called_once_with(
            api_keys_client._spaces_api, "my-space"
        )
        mock_api.list_api_keys.assert_called_once_with(
            key_type=None,
            status=None,
            space_id="resolved-space-id",
            user_id=None,
            limit=50,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from api_keys_list."""
        expected = Mock()
        mock_api.list_api_keys.return_value = expected

        result = api_keys_client.list()

        assert result is expected

    def test_list_emits_beta_prerelease_warning(
        self,
        api_keys_client: ApiKeysClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        api_keys_client.list()

        assert any(
            "BETA" in record.message and "api_keys.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestApiKeysClientCreate:
    """Tests for ApiKeysClient.create()."""

    def test_create_user_key_builds_request_and_calls_api(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create() should build CreateApiKeyRequest with key_type='USER'."""
        with patch(
            "arize._generated.api_client.CreateApiKeyRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            api_keys_client.create(name="my-key")

        mock_request_cls.assert_called_once_with(
            name="my-key",
            description=None,
            key_type="USER",
            expires_at=None,
        )
        mock_api.create_api_key.assert_called_once_with(
            create_api_key_request=mock_body
        )

    def test_create_with_all_params(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create() should pass all optional params to the request body."""
        expires = datetime(2030, 1, 1, tzinfo=timezone.utc)

        with patch(
            "arize._generated.api_client.CreateApiKeyRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            api_keys_client.create(
                name="my-key",
                description="A user key",
                expires_at=expires,
            )

        mock_request_cls.assert_called_once_with(
            name="my-key",
            description="A user key",
            key_type="USER",
            expires_at=expires,
        )
        mock_api.create_api_key.assert_called_once_with(
            create_api_key_request=mock_body
        )

    def test_create_returns_api_response(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from api_keys_create."""
        expected = Mock()
        mock_api.create_api_key.return_value = expected

        with patch("arize._generated.api_client.CreateApiKeyRequest"):
            result = api_keys_client.create(name="my-key")

        assert result is expected

    def test_create_emits_beta_prerelease_warning(
        self,
        api_keys_client: ApiKeysClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to create() should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch("arize._generated.api_client.CreateApiKeyRequest"):
            api_keys_client.create(name="my-key")

        assert any(
            "BETA" in record.message and "api_keys.create" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestApiKeysClientCreateServiceKey:
    """Tests for ApiKeysClient.create_service_key()."""

    # Base64 ID that decodes to "Space:905:abc" — passes is_resource_id()
    _SPACE_ID = "U3BhY2U6OTA1MDoxSmtS"

    def test_create_service_key_builds_request_and_calls_api(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should build CreateApiKeyRequest with key_type='SERVICE'."""
        with patch(
            "arize._generated.api_client.CreateApiKeyRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            api_keys_client.create_service_key(
                name="svc-key", space=self._SPACE_ID
            )

        mock_request_cls.assert_called_once_with(
            name="svc-key",
            description=None,
            key_type="SERVICE",
            expires_at=None,
            space_id=self._SPACE_ID,
            roles=None,
        )
        mock_api.create_api_key.assert_called_once_with(
            create_api_key_request=mock_body
        )

    def test_create_service_key_with_roles(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should build ApiKeyRoles when any role is set."""
        with (
            patch(
                "arize._generated.api_client.CreateApiKeyRequest"
            ) as mock_create_cls,
            patch("arize._generated.api_client.ApiKeyRoles") as mock_roles_cls,
        ):
            mock_roles = Mock()
            mock_roles_cls.return_value = mock_roles
            mock_create_cls.return_value = Mock()

            api_keys_client.create_service_key(
                name="svc-key",
                space=self._SPACE_ID,
                space_role="ADMIN",
                org_role="READ_ONLY",
                account_role="MEMBER",
            )

        mock_roles_cls.assert_called_once_with(
            space_role="ADMIN",
            org_role="READ_ONLY",
            account_role="MEMBER",
        )
        mock_create_cls.assert_called_once_with(
            name="svc-key",
            description=None,
            key_type="SERVICE",
            expires_at=None,
            space_id=self._SPACE_ID,
            roles=mock_roles,
        )

    def test_create_service_key_partial_roles(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should build ApiKeyRoles when only one role is set."""
        with (
            patch(
                "arize._generated.api_client.CreateApiKeyRequest"
            ) as mock_create_cls,
            patch("arize._generated.api_client.ApiKeyRoles") as mock_roles_cls,
        ):
            mock_roles_cls.return_value = Mock()
            mock_create_cls.return_value = Mock()

            api_keys_client.create_service_key(
                name="svc-key",
                space=self._SPACE_ID,
                space_role="READ_ONLY",
            )

        mock_roles_cls.assert_called_once_with(
            space_role="READ_ONLY",
            org_role=None,
            account_role=None,
        )

    def test_create_service_key_with_space_name_resolves_to_id(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should resolve a space name via _find_space_id."""
        with (
            patch(
                "arize.api_keys.client._find_space_id",
                return_value="resolved-space-id",
            ) as mock_resolve,
            patch(
                "arize._generated.api_client.CreateApiKeyRequest"
            ) as mock_request_cls,
        ):
            mock_request_cls.return_value = Mock()

            api_keys_client.create_service_key(name="svc-key", space="my-space")

        mock_resolve.assert_called_once_with(
            api_keys_client._spaces_api, "my-space"
        )
        mock_request_cls.assert_called_once_with(
            name="svc-key",
            description=None,
            key_type="SERVICE",
            expires_at=None,
            space_id="resolved-space-id",
            roles=None,
        )

    def test_create_service_key_with_all_params(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should pass description and expires_at through."""
        expires = datetime(2030, 1, 1, tzinfo=timezone.utc)

        with patch(
            "arize._generated.api_client.CreateApiKeyRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            api_keys_client.create_service_key(
                name="svc-key",
                space=self._SPACE_ID,
                description="My service key",
                expires_at=expires,
            )

        mock_request_cls.assert_called_once_with(
            name="svc-key",
            description="My service key",
            key_type="SERVICE",
            expires_at=expires,
            space_id=self._SPACE_ID,
            roles=None,
        )

    def test_create_service_key_returns_api_response(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should propagate the return value from api_keys_create."""
        expected = Mock()
        mock_api.create_api_key.return_value = expected

        with patch("arize._generated.api_client.CreateApiKeyRequest"):
            result = api_keys_client.create_service_key(
                name="svc-key", space=self._SPACE_ID
            )

        assert result is expected

    def test_create_service_key_emits_beta_prerelease_warning(
        self,
        api_keys_client: ApiKeysClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch("arize._generated.api_client.CreateApiKeyRequest"):
            api_keys_client.create_service_key(
                name="svc-key", space=self._SPACE_ID
            )

        assert any(
            "BETA" in record.message
            and "api_keys.create_service_key" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestApiKeysClientRevoke:
    """Tests for ApiKeysClient.revoke()."""

    def test_revoke_calls_api_with_key_id(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """revoke() should pass api_key_id to api_keys_revoke."""
        api_keys_client.revoke(api_key_id="key-123")

        mock_api.revoke_api_key.assert_called_once_with(api_key_id="key-123")

    def test_revoke_returns_none(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """revoke() should return None (204 No Content)."""
        mock_api.revoke_api_key.return_value = None

        result = api_keys_client.revoke(api_key_id="key-123")

        assert result is None

    def test_revoke_emits_beta_prerelease_warning(
        self,
        api_keys_client: ApiKeysClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to revoke() should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        api_keys_client.revoke(api_key_id="key-123")

        assert any(
            "BETA" in record.message and "api_keys.revoke" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestApiKeysClientRefresh:
    """Tests for ApiKeysClient.refresh()."""

    def test_refresh_calls_api_with_key_id(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """refresh() should build RefreshApiKeyRequest and call api_keys_refresh."""
        with patch(
            "arize._generated.api_client.RefreshApiKeyRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            api_keys_client.refresh(api_key_id="key-123")

        mock_request_cls.assert_called_once_with(
            expires_at=None, grace_period_seconds=None
        )
        mock_api.refresh_api_key.assert_called_once_with(
            api_key_id="key-123",
            refresh_api_key_request=mock_body,
        )

    def test_refresh_passes_expires_at(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """refresh() should forward expires_at to the request body."""
        expires = datetime(2030, 6, 1, tzinfo=timezone.utc)

        with patch(
            "arize._generated.api_client.RefreshApiKeyRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            api_keys_client.refresh(api_key_id="key-123", expires_at=expires)

        mock_request_cls.assert_called_once_with(
            expires_at=expires, grace_period_seconds=None
        )

    def test_refresh_passes_grace_period_seconds(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """refresh() should forward grace_period_seconds to request body."""
        with patch(
            "arize._generated.api_client.RefreshApiKeyRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            api_keys_client.refresh(
                api_key_id="key-123",
                grace_period_seconds=300,
            )

        mock_request_cls.assert_called_once_with(
            expires_at=None,
            grace_period_seconds=300,
        )

    def test_refresh_returns_api_response(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """refresh() should propagate the return value from api_keys_refresh."""
        expected = Mock()
        mock_api.refresh_api_key.return_value = expected

        with patch("arize._generated.api_client.RefreshApiKeyRequest"):
            result = api_keys_client.refresh(api_key_id="key-123")

        assert result is expected

    def test_refresh_emits_beta_prerelease_warning(
        self,
        api_keys_client: ApiKeysClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to refresh() should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch("arize._generated.api_client.RefreshApiKeyRequest"):
            api_keys_client.refresh(api_key_id="key-123")

        assert any(
            "BETA" in record.message and "api_keys.refresh" in record.message
            for record in caplog.records
        )
