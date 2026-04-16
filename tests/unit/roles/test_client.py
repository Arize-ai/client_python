"""Unit tests for src/arize/roles/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.roles.client import RolesClient

# Base64 ID that passes is_resource_id() — decodes to "Role:123"
_ROLE_ID = "Um9sZToxMjM="


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock RolesApi instance."""
    return Mock()


@pytest.fixture
def roles_client(mock_sdk_config: Mock, mock_api: Mock) -> RolesClient:
    """Provide a RolesClient with mocked internals."""
    with patch("arize._generated.api_client.RolesApi", return_value=mock_api):
        return RolesClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestRolesClientInit:
    """Tests for RolesClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.RolesApi", return_value=mock_api
        ):
            client = RolesClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_roles_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to RolesApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.RolesApi"
        ) as mock_roles_api_cls:
            RolesClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_roles_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestRolesClientList:
    """Tests for RolesClient.list()."""

    def test_list_calls_api_with_all_params(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """list() should pass limit, cursor, and is_predefined to roles_list."""
        roles_client.list(
            limit=50,
            cursor="cursor-abc",
            is_predefined=True,
        )

        mock_api.roles_list.assert_called_once_with(
            limit=50,
            cursor="cursor-abc",
            is_predefined=True,
        )

    def test_list_defaults(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """list() should default cursor/is_predefined to None and limit to 100."""
        roles_client.list()

        mock_api.roles_list.assert_called_once_with(
            limit=100,
            cursor=None,
            is_predefined=None,
        )

    def test_list_returns_api_response(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from roles_list."""
        expected = Mock()
        mock_api.roles_list.return_value = expected

        result = roles_client.list()

        assert result is expected

    def test_list_emits_alpha_prerelease_warning(
        self,
        roles_client: RolesClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        roles_client.list()

        assert any(
            "ALPHA" in record.message and "roles.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestRolesClientGet:
    """Tests for RolesClient.get()."""

    def test_get_with_id_calls_api_directly(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """get() with a base64 ID should pass it directly to roles_get (no list call)."""
        roles_client.get(role=_ROLE_ID)

        mock_api.roles_get.assert_called_once_with(role_id=_ROLE_ID)
        mock_api.roles_list.assert_not_called()

    def test_get_with_name_resolves_via_roles_list(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """get() with a name should paginate roles_list to find the matching ID."""
        mock_role = Mock()
        mock_role.id = _ROLE_ID
        mock_role.name = "My Role"
        mock_api.roles_list.return_value = Mock(
            roles=[mock_role],
            pagination=Mock(next_cursor=None),
        )

        roles_client.get(role="My Role")

        mock_api.roles_list.assert_called_once_with(limit=100, cursor=None)
        mock_api.roles_get.assert_called_once_with(role_id=_ROLE_ID)

    def test_get_with_name_not_found_raises(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """get() should raise NotFoundError when the role name is not found."""
        from arize.utils.resolve import NotFoundError

        mock_api.roles_list.return_value = Mock(
            roles=[],
            pagination=Mock(next_cursor=None),
        )

        with pytest.raises(NotFoundError, match="role"):
            roles_client.get(role="nonexistent-role")

    def test_get_returns_api_response(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """get() should propagate the return value from roles_get."""
        expected = Mock()
        mock_api.roles_get.return_value = expected

        result = roles_client.get(role=_ROLE_ID)

        assert result is expected

    def test_get_emits_alpha_prerelease_warning(
        self,
        roles_client: RolesClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        roles_client.get(role=_ROLE_ID)

        assert any(
            "ALPHA" in record.message and "roles.get" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestRolesClientCreate:
    """Tests for RolesClient.create()."""

    def test_create_builds_request_and_calls_api(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """create() should build RoleCreate and pass it to roles_create."""
        with patch(
            "arize._generated.api_client.RoleCreate"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            roles_client.create(
                name="Data Scientist",
                permissions=["PROJECT_READ", "DATASET_CREATE"],
                description="Can read projects and create datasets.",
            )

        mock_request_cls.assert_called_once_with(
            name="Data Scientist",
            permissions=["PROJECT_READ", "DATASET_CREATE"],
            description="Can read projects and create datasets.",
        )
        mock_api.roles_create.assert_called_once_with(role_create=mock_body)

    def test_create_without_description(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """create() should pass description=None when not provided."""
        with patch(
            "arize._generated.api_client.RoleCreate"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            roles_client.create(
                name="Viewer",
                permissions=["PROJECT_READ"],
            )

        mock_request_cls.assert_called_once_with(
            name="Viewer",
            permissions=["PROJECT_READ"],
            description=None,
        )

    def test_create_returns_api_response(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from roles_create."""
        expected = Mock()
        mock_api.roles_create.return_value = expected

        with patch("arize._generated.api_client.RoleCreate"):
            result = roles_client.create(
                name="Analyst",
                permissions=["PROJECT_READ"],
            )

        assert result is expected


@pytest.mark.unit
class TestRolesClientUpdate:
    """Tests for RolesClient.update()."""

    def test_update_raises_when_no_fields_provided(
        self, roles_client: RolesClient
    ) -> None:
        """update() should raise if none of name, description, or permissions is provided."""
        with pytest.raises(
            ValueError,
            match="At least one of 'name', 'description', or 'permissions' must be provided",
        ):
            roles_client.update(role=_ROLE_ID)

    def test_update_builds_request_and_calls_api(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """update() should build RoleUpdate and pass it to roles_update."""
        with patch(
            "arize._generated.api_client.RoleUpdate"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            roles_client.update(
                role=_ROLE_ID,
                name="Senior Analyst",
                description="Updated description",
                permissions=["PROJECT_READ", "DATASET_READ"],
            )

        mock_request_cls.assert_called_once_with(
            name="Senior Analyst",
            description="Updated description",
            permissions=["PROJECT_READ", "DATASET_READ"],
        )
        mock_api.roles_update.assert_called_once_with(
            role_id=_ROLE_ID,
            role_update=mock_body,
        )

    def test_update_with_only_name(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """update() with only name should pass None for other fields."""
        with patch(
            "arize._generated.api_client.RoleUpdate"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            roles_client.update(role=_ROLE_ID, name="New Name")

        mock_request_cls.assert_called_once_with(
            name="New Name",
            description=None,
            permissions=None,
        )

    def test_update_returns_api_response(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """update() should propagate the return value from roles_update."""
        expected = Mock()
        mock_api.roles_update.return_value = expected

        with patch("arize._generated.api_client.RoleUpdate"):
            result = roles_client.update(
                role=_ROLE_ID,
                name="Updated Name",
            )

        assert result is expected


@pytest.mark.unit
class TestRolesClientDelete:
    """Tests for RolesClient.delete()."""

    def test_delete_with_id_calls_api_directly(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """delete() with a base64 ID should pass it directly to roles_delete (no list call)."""
        roles_client.delete(role=_ROLE_ID)

        mock_api.roles_delete.assert_called_once_with(role_id=_ROLE_ID)
        mock_api.roles_list.assert_not_called()

    def test_delete_with_name_resolves_via_roles_list(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """delete() with a name should paginate roles_list to find the matching ID."""
        mock_role = Mock()
        mock_role.id = _ROLE_ID
        mock_role.name = "My Role"
        mock_api.roles_list.return_value = Mock(
            roles=[mock_role],
            pagination=Mock(next_cursor=None),
        )

        roles_client.delete(role="My Role")

        mock_api.roles_list.assert_called_once_with(limit=100, cursor=None)
        mock_api.roles_delete.assert_called_once_with(role_id=_ROLE_ID)

    def test_delete_with_name_not_found_raises(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """delete() should raise NotFoundError when the role name is not found."""
        from arize.utils.resolve import NotFoundError

        mock_api.roles_list.return_value = Mock(
            roles=[],
            pagination=Mock(next_cursor=None),
        )

        with pytest.raises(NotFoundError, match="role"):
            roles_client.delete(role="nonexistent-role")

    def test_delete_returns_none(
        self, roles_client: RolesClient, mock_api: Mock
    ) -> None:
        """delete() should return None (204 No Content)."""
        mock_api.roles_delete.return_value = None

        result = roles_client.delete(role=_ROLE_ID)

        assert result is None
