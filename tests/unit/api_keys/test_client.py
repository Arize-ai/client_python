"""Unit tests for src/arize/api_keys/client.py."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from arize._generated.api_client import ApiKeyStatus, ApiKeyType
from arize.api_keys.client import ApiKeysClient
from arize.api_keys.types import OrgBinding, SpaceBinding


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

    # Base64 ID that decodes to "Space:905:abc" — passes is_resource_id()
    _SPACE_ID = "U3BhY2U6OTA1MDoxSmtS"

    def test_list_calls_api_with_all_params(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """list() should forward all parameters to api_keys_list."""
        api_keys_client.list(
            key_type=ApiKeyType.SERVICE,
            status=ApiKeyStatus.ACTIVE,
            space=self._SPACE_ID,
            user_id="VXNlcjoxMjM0NQ==",
            limit=25,
            cursor="cursor-abc",
        )

        mock_api.list_api_keys.assert_called_once_with(
            key_type=ApiKeyType.SERVICE,
            status=ApiKeyStatus.ACTIVE,
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
        """create() should build UserApiKeyCreate wrapped in ApiKeyCreate."""
        mock_user_key = Mock()
        mock_body = Mock()

        with (
            patch(
                "arize._generated.api_client.CreateUserApiKeyRequest",
                return_value=mock_user_key,
            ) as mock_user_cls,
            patch(
                "arize._generated.api_client.CreateApiKeyRequest",
                return_value=mock_body,
            ) as mock_create_cls,
        ):
            api_keys_client.create(name="my-key")

        mock_user_cls.assert_called_once_with(
            key_type="USER",
            name="my-key",
            description=None,
            expires_at=None,
        )
        mock_create_cls.assert_called_once_with(mock_user_key)
        mock_api.create_api_key.assert_called_once_with(
            create_api_key_request=mock_body
        )

    def test_create_with_all_params(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create() should pass all optional params to UserApiKeyCreate."""
        expires = datetime(2030, 1, 1, tzinfo=timezone.utc)

        with (
            patch(
                "arize._generated.api_client.CreateUserApiKeyRequest",
            ) as mock_user_cls,
            patch("arize._generated.api_client.CreateApiKeyRequest"),
        ):
            api_keys_client.create(
                name="my-key",
                description="A user key",
                expires_at=expires,
            )

        mock_user_cls.assert_called_once_with(
            key_type="USER",
            name="my-key",
            description="A user key",
            expires_at=expires,
        )

    def test_create_returns_api_response(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create() should unwrap the generated oneOf response."""
        expected = Mock()
        mock_api.create_api_key.return_value = Mock(actual_instance=expected)

        with (
            patch("arize._generated.api_client.CreateUserApiKeyRequest"),
            patch("arize._generated.api_client.CreateApiKeyRequest"),
        ):
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

        with (
            patch("arize._generated.api_client.CreateUserApiKeyRequest"),
            patch("arize._generated.api_client.CreateApiKeyRequest"),
        ):
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
    _ORG_ID = "T3JnMTIz"

    def test_create_service_key_builds_request_and_calls_api(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should build the nested orgs/spaces structure."""
        with (
            patch(
                "arize._generated.api_client.ServiceKeySpaceAssignment"
            ) as mock_space_cls,
            patch(
                "arize._generated.api_client.ServiceKeyOrgAssignment"
            ) as mock_org_cls,
            patch(
                "arize._generated.api_client.CreateServiceApiKeyRequest"
            ) as mock_svc_cls,
            patch(
                "arize._generated.api_client.CreateApiKeyRequest"
            ) as mock_body_cls,
        ):
            mock_space_binding = Mock()
            mock_space_cls.return_value = mock_space_binding
            mock_org_binding = Mock()
            mock_org_cls.return_value = mock_org_binding
            mock_svc_body = Mock()
            mock_svc_cls.return_value = mock_svc_body
            mock_body = Mock()
            mock_body_cls.return_value = mock_body

            api_keys_client.create_service_key(
                name="svc-key",
                orgs=[
                    OrgBinding(
                        org_id=self._ORG_ID,
                        spaces=[SpaceBinding(space=self._SPACE_ID)],
                    )
                ],
            )

        mock_space_cls.assert_called_once_with(
            space_id=self._SPACE_ID,
            role=None,
        )
        mock_org_cls.assert_called_once_with(
            org_id=self._ORG_ID,
            role=None,
            spaces=[mock_space_binding],
        )
        mock_svc_cls.assert_called_once_with(
            key_type="SERVICE",
            name="svc-key",
            description=None,
            expires_at=None,
            account_role=None,
            organizations=[mock_org_binding],
        )
        mock_body_cls.assert_called_once_with(mock_svc_body)
        mock_api.create_api_key.assert_called_once_with(
            create_api_key_request=mock_body
        )

    def test_create_service_key_omitted_roles_are_absent_from_payload(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """Omitted roles should be absent so the server can apply defaults."""
        api_keys_client.create_service_key(
            name="svc-key",
            orgs=[
                OrgBinding(
                    org_id=self._ORG_ID,
                    spaces=[SpaceBinding(space=self._SPACE_ID)],
                )
            ],
        )

        body = mock_api.create_api_key.call_args.kwargs[
            "create_api_key_request"
        ]
        serialized = body.to_dict()
        org = serialized["organizations"][0]
        space = org["spaces"][0]

        assert "account_role" not in serialized
        assert "role" not in org
        assert "role" not in space

    def test_create_service_key_multi_space(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should resolve and build all space bindings."""
        space_id_2 = "U3BhY2U6NDU2OmRlZg=="
        mock_role = Mock()

        with (
            patch(
                "arize.api_keys.client._find_space_id",
                side_effect=[self._SPACE_ID, space_id_2],
            ),
            patch(
                "arize._generated.api_client.ServiceKeySpaceAssignment"
            ) as mock_binding_cls,
            patch("arize._generated.api_client.ServiceKeyOrgAssignment"),
            patch("arize._generated.api_client.CreateServiceApiKeyRequest"),
            patch("arize._generated.api_client.CreateApiKeyRequest"),
        ):
            mock_binding_cls.side_effect = [Mock(), Mock()]

            api_keys_client.create_service_key(
                name="svc-key",
                orgs=[
                    OrgBinding(
                        org_id=self._ORG_ID,
                        spaces=[
                            SpaceBinding(space="space-one"),
                            SpaceBinding(space=space_id_2, role=mock_role),
                        ],
                    )
                ],
            )

        assert mock_binding_cls.call_count == 2
        mock_binding_cls.assert_any_call(space_id=self._SPACE_ID, role=None)
        mock_binding_cls.assert_any_call(space_id=space_id_2, role=mock_role)

    def test_create_service_key_with_space_name_resolves_to_id(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should resolve a space name via _find_space_id."""
        with (
            patch(
                "arize.api_keys.client._find_space_id",
                return_value="resolved-space-id",
            ) as mock_resolve,
            patch("arize._generated.api_client.ServiceKeySpaceAssignment"),
            patch("arize._generated.api_client.ServiceKeyOrgAssignment"),
            patch("arize._generated.api_client.CreateServiceApiKeyRequest"),
            patch("arize._generated.api_client.CreateApiKeyRequest"),
        ):
            api_keys_client.create_service_key(
                name="svc-key",
                orgs=[
                    OrgBinding(
                        org_id=self._ORG_ID,
                        spaces=[SpaceBinding(space="my-space")],
                    )
                ],
            )

        mock_resolve.assert_called_once_with(
            api_keys_client._spaces_api, "my-space"
        )

    def test_create_service_key_with_all_params(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should pass description, expires_at, and account_role."""
        expires = datetime(2030, 1, 1, tzinfo=timezone.utc)
        mock_account_role = Mock()

        with (
            patch("arize._generated.api_client.ServiceKeySpaceAssignment"),
            patch(
                "arize._generated.api_client.ServiceKeyOrgAssignment"
            ) as mock_org_cls,
            patch(
                "arize._generated.api_client.CreateServiceApiKeyRequest"
            ) as mock_svc_cls,
            patch("arize._generated.api_client.CreateApiKeyRequest"),
        ):
            mock_org_binding = Mock()
            mock_org_cls.return_value = mock_org_binding
            api_keys_client.create_service_key(
                name="svc-key",
                orgs=[
                    OrgBinding(
                        org_id=self._ORG_ID,
                        spaces=[SpaceBinding(space=self._SPACE_ID)],
                    )
                ],
                account_role=mock_account_role,
                description="My service key",
                expires_at=expires,
            )

        mock_svc_cls.assert_called_once_with(
            key_type="SERVICE",
            name="svc-key",
            description="My service key",
            expires_at=expires,
            account_role=mock_account_role,
            organizations=[mock_org_binding],
        )

    def test_create_service_key_raises_on_empty_orgs(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should raise ValueError when orgs is empty."""
        with pytest.raises(ValueError, match="at least one entry"):
            api_keys_client.create_service_key(name="svc-key", orgs=[])

    def test_create_service_key_raises_on_empty_spaces_in_org(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should raise ValueError when an org has no spaces."""
        with pytest.raises(ValueError, match="at least one SpaceBinding"):
            api_keys_client.create_service_key(
                name="svc-key",
                orgs=[OrgBinding(org_id=self._ORG_ID, spaces=[])],
            )

    def test_create_service_key_returns_api_response(
        self, api_keys_client: ApiKeysClient, mock_api: Mock
    ) -> None:
        """create_service_key() should unwrap the generated oneOf response."""
        expected = Mock()
        mock_api.create_api_key.return_value = Mock(actual_instance=expected)

        with (
            patch("arize._generated.api_client.ServiceKeySpaceAssignment"),
            patch("arize._generated.api_client.ServiceKeyOrgAssignment"),
            patch("arize._generated.api_client.CreateServiceApiKeyRequest"),
            patch("arize._generated.api_client.CreateApiKeyRequest"),
        ):
            result = api_keys_client.create_service_key(
                name="svc-key",
                orgs=[
                    OrgBinding(
                        org_id=self._ORG_ID,
                        spaces=[SpaceBinding(space=self._SPACE_ID)],
                    )
                ],
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

        with (
            patch("arize._generated.api_client.ServiceKeySpaceAssignment"),
            patch("arize._generated.api_client.ServiceKeyOrgAssignment"),
            patch("arize._generated.api_client.CreateServiceApiKeyRequest"),
            patch("arize._generated.api_client.CreateApiKeyRequest"),
        ):
            api_keys_client.create_service_key(
                name="svc-key",
                orgs=[
                    OrgBinding(
                        org_id=self._ORG_ID,
                        spaces=[SpaceBinding(space=self._SPACE_ID)],
                    )
                ],
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
        """refresh() should build ApiKeyRefresh and call api_keys_refresh."""
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
