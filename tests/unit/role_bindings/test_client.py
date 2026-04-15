"""Unit tests for src/arize/role_bindings/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.role_bindings.client import RoleBindingsClient


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock RoleBindingsApi instance."""
    return Mock()


@pytest.fixture
def role_bindings_client(
    mock_sdk_config: Mock, mock_api: Mock
) -> RoleBindingsClient:
    """Provide a RoleBindingsClient with mocked internals."""
    with patch(
        "arize._generated.api_client.RoleBindingsApi", return_value=mock_api
    ):
        return RoleBindingsClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestRoleBindingsClientInit:
    """Tests for RoleBindingsClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.RoleBindingsApi", return_value=mock_api
        ):
            client = RoleBindingsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_role_bindings_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to RoleBindingsApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.RoleBindingsApi"
        ) as mock_api_cls:
            RoleBindingsClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestRoleBindingsClientCreate:
    """Tests for RoleBindingsClient.create()."""

    def test_create_builds_request_and_calls_api(
        self, role_bindings_client: RoleBindingsClient, mock_api: Mock
    ) -> None:
        """create() should build RoleBindingCreate and pass it to role_bindings_create."""
        with patch(
            "arize._generated.api_client.RoleBindingCreate"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            role_bindings_client.create(
                user_id="user-123",
                role_id="role-456",
                resource_type="PROJECT",
                resource_id="project-789",
            )

        mock_request_cls.assert_called_once_with(
            user_id="user-123",
            role_id="role-456",
            resource_type="PROJECT",
            resource_id="project-789",
        )
        mock_api.role_bindings_create.assert_called_once_with(mock_body)

    def test_create_returns_api_response(
        self, role_bindings_client: RoleBindingsClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from role_bindings_create."""
        expected = Mock()
        mock_api.role_bindings_create.return_value = expected

        with patch("arize._generated.api_client.RoleBindingCreate"):
            result = role_bindings_client.create(
                user_id="user-123",
                role_id="role-456",
                resource_type="PROJECT",
                resource_id="project-789",
            )

        assert result is expected

    def test_create_emits_alpha_prerelease_warning(
        self,
        role_bindings_client: RoleBindingsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch("arize._generated.api_client.RoleBindingCreate"):
            role_bindings_client.create(
                user_id="user-123",
                role_id="role-456",
                resource_type="PROJECT",
                resource_id="project-789",
            )

        assert any(
            "ALPHA" in record.message
            and "role_bindings.create" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestRoleBindingsClientGet:
    """Tests for RoleBindingsClient.get()."""

    def test_get_calls_api_with_binding_id(
        self, role_bindings_client: RoleBindingsClient, mock_api: Mock
    ) -> None:
        """get() should pass binding_id to role_bindings_get."""
        role_bindings_client.get(binding_id="binding-123")

        mock_api.role_bindings_get.assert_called_once_with("binding-123")

    def test_get_returns_api_response(
        self, role_bindings_client: RoleBindingsClient, mock_api: Mock
    ) -> None:
        """get() should propagate the return value from role_bindings_get."""
        expected = Mock()
        mock_api.role_bindings_get.return_value = expected

        result = role_bindings_client.get(binding_id="binding-123")

        assert result is expected


@pytest.mark.unit
class TestRoleBindingsClientUpdate:
    """Tests for RoleBindingsClient.update()."""

    def test_update_builds_request_and_calls_api(
        self, role_bindings_client: RoleBindingsClient, mock_api: Mock
    ) -> None:
        """update() should build RoleBindingUpdate and pass it to role_bindings_update."""
        with patch(
            "arize._generated.api_client.RoleBindingUpdate"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            role_bindings_client.update(
                binding_id="binding-123",
                role_id="role-new",
            )

        mock_request_cls.assert_called_once_with(role_id="role-new")
        mock_api.role_bindings_update.assert_called_once_with(
            "binding-123", mock_body
        )

    def test_update_returns_api_response(
        self, role_bindings_client: RoleBindingsClient, mock_api: Mock
    ) -> None:
        """update() should propagate the return value from role_bindings_update."""
        expected = Mock()
        mock_api.role_bindings_update.return_value = expected

        with patch("arize._generated.api_client.RoleBindingUpdate"):
            result = role_bindings_client.update(
                binding_id="binding-123",
                role_id="role-new",
            )

        assert result is expected


@pytest.mark.unit
class TestRoleBindingsClientDelete:
    """Tests for RoleBindingsClient.delete()."""

    def test_delete_calls_api_with_binding_id(
        self, role_bindings_client: RoleBindingsClient, mock_api: Mock
    ) -> None:
        """delete() should pass binding_id to role_bindings_delete."""
        role_bindings_client.delete(binding_id="binding-123")

        mock_api.role_bindings_delete.assert_called_once_with("binding-123")

    def test_delete_returns_none(
        self, role_bindings_client: RoleBindingsClient, mock_api: Mock
    ) -> None:
        """delete() should return None (204 No Content)."""
        mock_api.role_bindings_delete.return_value = None

        result = role_bindings_client.delete(binding_id="binding-123")

        assert result is None
