"""Unit tests for src/arize/api_keys/client.py."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from arize.api_keys.client import ApiKeysClient


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock ApiKeysApi instance."""
    return Mock()


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

    def test_list_calls_api_with_all_params(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """list() should forward all parameters to api_keys_list."""
        api_keys_client.list(
            key_type="service",
            status="active",
            limit=25,
            cursor="cursor-abc",
        )

        mock_api.api_keys_list.assert_called_once_with(
            key_type="service",
            status="active",
            limit=25,
            cursor="cursor-abc",
        )

    def test_list_defaults(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """list() should default key_type/status/cursor to None and limit to 50."""
        api_keys_client.list()

        mock_api.api_keys_list.assert_called_once_with(
            key_type=None,
            status=None,
            limit=50,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from api_keys_list."""
        expected = Mock()
        mock_api.api_keys_list.return_value = expected

        result = api_keys_client.list()

        assert result is expected

    def test_list_emits_alpha_prerelease_warning(
        self,
        api_keys_client: ApiKeysClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        api_keys_client.list()

        assert any(
            "ALPHA" in record.message and "api_keys.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestApiKeysClientCreate:
    """Tests for ApiKeysClient.create()."""

    # Base64 ID that decodes to "Space:905:abc" — passes is_resource_id()
    _SPACE_ID = "U3BhY2U6OTA1MDoxSmtS"

    def test_create_user_key_builds_request_and_calls_api(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create() should build ApiKeyCreate and pass it to api_keys_create."""
        with patch(
            "arize._generated.api_client.ApiKeyCreate"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            api_keys_client.create(name="my-key")

        mock_request_cls.assert_called_once_with(
            name="my-key",
            description=None,
            key_type="user",
            expires_at=None,
            space_id=None,
        )
        mock_api.api_keys_create.assert_called_once_with(
            api_key_create=mock_body
        )

    def test_create_with_all_params(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create() should pass all optional params to the request body."""
        expires = datetime(2030, 1, 1, tzinfo=timezone.utc)

        with patch(
            "arize._generated.api_client.ApiKeyCreate"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            api_keys_client.create(
                name="svc-key",
                description="A service key",
                key_type="service",
                expires_at=expires,
                space=self._SPACE_ID,
            )

        mock_request_cls.assert_called_once_with(
            name="svc-key",
            description="A service key",
            key_type="service",
            expires_at=expires,
            space_id=self._SPACE_ID,
        )
        mock_api.api_keys_create.assert_called_once_with(
            api_key_create=mock_body
        )

    def test_create_with_space_name_resolves_to_id(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create() should resolve a space name to an ID via _find_space_id."""
        with (
            patch(
                "arize.api_keys.client._find_space_id",
                return_value="resolved-space-id",
            ) as mock_resolve,
            patch(
                "arize._generated.api_client.ApiKeyCreate"
            ) as mock_request_cls,
        ):
            mock_request_cls.return_value = Mock()

            api_keys_client.create(
                name="svc-key",
                key_type="service",
                space="my-space",
            )

        mock_resolve.assert_called_once_with(
            api_keys_client._spaces_api, "my-space"
        )
        mock_request_cls.assert_called_once_with(
            name="svc-key",
            description=None,
            key_type="service",
            expires_at=None,
            space_id="resolved-space-id",
        )

    def test_create_returns_api_response(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from api_keys_create."""
        expected = Mock()
        mock_api.api_keys_create.return_value = expected

        with patch("arize._generated.api_client.ApiKeyCreate"):
            result = api_keys_client.create(name="my-key")

        assert result is expected

    def test_create_emits_alpha_prerelease_warning(
        self,
        api_keys_client: ApiKeysClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to create() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch("arize._generated.api_client.ApiKeyCreate"):
            api_keys_client.create(name="my-key")

        assert any(
            "ALPHA" in record.message and "api_keys.create" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestApiKeysClientDelete:
    """Tests for ApiKeysClient.delete()."""

    def test_delete_calls_api_with_key_id(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """delete() should pass api_key_id to api_keys_delete."""
        api_keys_client.delete(api_key_id="key-123")

        mock_api.api_keys_delete.assert_called_once_with(api_key_id="key-123")

    def test_delete_returns_none(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """delete() should return None (204 No Content)."""
        mock_api.api_keys_delete.return_value = None

        result = api_keys_client.delete(api_key_id="key-123")

        assert result is None

    def test_delete_emits_alpha_prerelease_warning(
        self,
        api_keys_client: ApiKeysClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to delete() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        api_keys_client.delete(api_key_id="key-123")

        assert any(
            "ALPHA" in record.message and "api_keys.delete" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestApiKeysClientRefresh:
    """Tests for ApiKeysClient.refresh()."""

    def test_refresh_calls_api_with_key_id(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """refresh() should build ApiKeyRefresh and call api_keys_refresh."""
        with patch(
            "arize._generated.api_client.ApiKeyRefresh"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            api_keys_client.refresh(api_key_id="key-123")

        mock_request_cls.assert_called_once_with(expires_at=None)
        mock_api.api_keys_refresh.assert_called_once_with(
            api_key_id="key-123",
            api_key_refresh=mock_body,
        )

    def test_refresh_passes_expires_at(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """refresh() should forward expires_at to the request body."""
        expires = datetime(2030, 6, 1, tzinfo=timezone.utc)

        with patch(
            "arize._generated.api_client.ApiKeyRefresh"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            api_keys_client.refresh(api_key_id="key-123", expires_at=expires)

        mock_request_cls.assert_called_once_with(expires_at=expires)

    def test_refresh_returns_api_response(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """refresh() should propagate the return value from api_keys_refresh."""
        expected = Mock()
        mock_api.api_keys_refresh.return_value = expected

        with patch("arize._generated.api_client.ApiKeyRefresh"):
            result = api_keys_client.refresh(api_key_id="key-123")

        assert result is expected

    def test_refresh_emits_alpha_prerelease_warning(
        self,
        api_keys_client: ApiKeysClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to refresh() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch("arize._generated.api_client.ApiKeyRefresh"):
            api_keys_client.refresh(api_key_id="key-123")

        assert any(
            "ALPHA" in record.message and "api_keys.refresh" in record.message
            for record in caplog.records
        )
