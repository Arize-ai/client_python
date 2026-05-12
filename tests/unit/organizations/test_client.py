"""Unit tests for src/arize/organizations/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.organizations.client import OrganizationsClient
from arize.organizations.types import OrganizationRole, PredefinedOrgRole


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock OrganizationsApi instance."""
    return Mock()


@pytest.fixture
def organizations_client(
    mock_sdk_config: Mock, mock_api: Mock
) -> OrganizationsClient:
    """Provide an OrganizationsClient with mocked internals."""
    with patch(
        "arize._generated.api_client.OrganizationsApi", return_value=mock_api
    ):
        return OrganizationsClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestOrganizationsClientInit:
    """Tests for OrganizationsClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.OrganizationsApi",
            return_value=mock_api,
        ):
            client = OrganizationsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_organizations_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to OrganizationsApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.OrganizationsApi"
        ) as mock_orgs_api_cls:
            OrganizationsClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_orgs_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestOrganizationsClientList:
    """Tests for OrganizationsClient.list()."""

    def test_list_calls_api_with_all_params(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """list() should pass name, limit, and cursor to organizations_list."""
        organizations_client.list(
            name="my-org",
            limit=25,
            cursor="cursor-abc",
        )

        mock_api.organizations_list.assert_called_once_with(
            name="my-org",
            limit=25,
            cursor="cursor-abc",
        )

    def test_list_defaults(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """list() should default name/cursor to None and limit to 50."""
        organizations_client.list()

        mock_api.organizations_list.assert_called_once_with(
            name=None,
            limit=50,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from organizations_list."""
        expected = Mock()
        mock_api.organizations_list.return_value = expected

        result = organizations_client.list()

        assert result is expected

    def test_list_emits_alpha_prerelease_warning(
        self,
        organizations_client: OrganizationsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        organizations_client.list()

        assert any(
            "ALPHA" in record.message and "organizations.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestOrganizationsClientGet:
    """Tests for OrganizationsClient.get()."""

    def test_get_calls_api_with_org_id(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """get() should resolve organization and pass org_id to organizations_get."""
        organizations_client.get(organization="T3JnYW5pemF0aW9uOjEyMzQ1")

        mock_api.organizations_get.assert_called_once_with(
            org_id="T3JnYW5pemF0aW9uOjEyMzQ1"
        )

    def test_get_returns_api_response(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """get() should propagate the return value from organizations_get."""
        expected = Mock()
        mock_api.organizations_get.return_value = expected

        result = organizations_client.get(
            organization="T3JnYW5pemF0aW9uOjEyMzQ1"
        )

        assert result is expected


@pytest.mark.unit
class TestOrganizationsClientCreate:
    """Tests for OrganizationsClient.create()."""

    def test_create_builds_request_and_calls_api(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """create() should build OrganizationsCreateRequest and pass it to organizations_create."""
        with patch(
            "arize._generated.api_client.OrganizationsCreateRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            organizations_client.create(
                name="my-org",
                description="my description",
            )

        mock_request_cls.assert_called_once_with(
            name="my-org",
            description="my description",
        )
        mock_api.organizations_create.assert_called_once_with(
            organizations_create_request=mock_body
        )

    def test_create_returns_api_response(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from organizations_create."""
        expected = Mock()
        mock_api.organizations_create.return_value = expected

        with patch("arize._generated.api_client.OrganizationsCreateRequest"):
            result = organizations_client.create(name="my-org")

        assert result is expected


@pytest.mark.unit
class TestOrganizationsClientUpdate:
    """Tests for OrganizationsClient.update()."""

    def test_update_raises_when_no_fields_provided(
        self, organizations_client: OrganizationsClient
    ) -> None:
        """update() should raise if neither name nor description is provided."""
        with pytest.raises(
            ValueError,
            match="At least one of 'name' or 'description' must be provided",
        ):
            organizations_client.update(organization="T3JnYW5pemF0aW9uOjEyMzQ1")

    def test_update_builds_request_and_calls_api(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """update() should build OrganizationsUpdateRequest and pass it to organizations_update."""
        with patch(
            "arize._generated.api_client.OrganizationsUpdateRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            organizations_client.update(
                organization="T3JnYW5pemF0aW9uOjEyMzQ1",
                name="updated-org",
                description="updated description",
            )

        mock_request_cls.assert_called_once_with(
            name="updated-org",
            description="updated description",
        )
        mock_api.organizations_update.assert_called_once_with(
            org_id="T3JnYW5pemF0aW9uOjEyMzQ1",
            organizations_update_request=mock_body,
        )

    def test_update_returns_api_response(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """update() should propagate the return value from organizations_update."""
        expected = Mock()
        mock_api.organizations_update.return_value = expected

        with patch("arize._generated.api_client.OrganizationsUpdateRequest"):
            result = organizations_client.update(
                organization="T3JnYW5pemF0aW9uOjEyMzQ1",
                name="updated-org",
            )

        assert result is expected


@pytest.mark.unit
class TestOrganizationsClientDelete:
    """Tests for OrganizationsClient.delete()."""

    def test_delete_calls_api_with_org_id(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """delete() should resolve organization and pass org_id to organizations_delete."""
        organizations_client.delete(organization="T3JnYW5pemF0aW9uOjEyMzQ1")

        mock_api.organizations_delete.assert_called_once_with(
            org_id="T3JnYW5pemF0aW9uOjEyMzQ1"
        )

    def test_delete_returns_none(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """delete() should return None on success (204 response)."""
        mock_api.organizations_delete.return_value = None

        result = organizations_client.delete(
            organization="T3JnYW5pemF0aW9uOjEyMzQ1"
        )

        assert result is None

    def test_delete_emits_alpha_prerelease_warning(
        self,
        organizations_client: OrganizationsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        organizations_client.delete(organization="T3JnYW5pemF0aW9uOjEyMzQ1")

        assert any(
            "ALPHA" in record.message
            and "organizations.delete" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestOrganizationsClientAddUser:
    """Tests for OrganizationsClient.add_user()."""

    def test_add_user_with_predefined_role_calls_api(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """add_user() with PredefinedOrgRole should call _to_generated() and wrap in OrganizationRoleAssignment."""
        role = PredefinedOrgRole(name=OrganizationRole.MEMBER)
        with (
            patch(
                "arize._generated.api_client.OrganizationMembershipInput"
            ) as mock_input_cls,
            patch(
                "arize._generated.api_client.OrganizationRoleAssignment"
            ) as mock_role_cls,
        ):
            mock_body = Mock()
            mock_input_cls.return_value = mock_body
            mock_role = Mock()
            mock_role_cls.return_value = mock_role

            organizations_client.add_user(
                organization="T3JnYW5pemF0aW9uOjEyMzQ1",
                user_id="VXNlcjoxMjM0NQ==",
                role=role,
            )

        mock_role_cls.assert_called_once_with(role._to_generated())
        mock_input_cls.assert_called_once_with(
            user_id="VXNlcjoxMjM0NQ==",
            role=mock_role,
        )
        mock_api.organizations_add_user.assert_called_once_with(
            org_id="T3JnYW5pemF0aW9uOjEyMzQ1",
            organization_membership_input=mock_body,
        )

    def test_add_user_returns_api_response(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """add_user() should propagate the return value from organizations_add_user."""
        expected = Mock()
        mock_api.organizations_add_user.return_value = expected

        with (
            patch("arize._generated.api_client.OrganizationMembershipInput"),
            patch("arize._generated.api_client.OrganizationRoleAssignment"),
        ):
            result = organizations_client.add_user(
                organization="T3JnYW5pemF0aW9uOjEyMzQ1",
                user_id="VXNlcjoxMjM0NQ==",
                role=PredefinedOrgRole(name=OrganizationRole.MEMBER),
            )

        assert result is expected

    def test_add_user_emits_alpha_prerelease_warning(
        self,
        organizations_client: OrganizationsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch("arize._generated.api_client.OrganizationMembershipInput"),
            patch("arize._generated.api_client.OrganizationRoleAssignment"),
        ):
            organizations_client.add_user(
                organization="T3JnYW5pemF0aW9uOjEyMzQ1",
                user_id="VXNlcjoxMjM0NQ==",
                role=PredefinedOrgRole(name=OrganizationRole.MEMBER),
            )

        assert any(
            "ALPHA" in record.message
            and "organizations.add_user" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestOrganizationsClientRemoveUser:
    """Tests for OrganizationsClient.remove_user()."""

    def test_remove_user_calls_api_with_org_and_user_id(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """remove_user() should resolve organization and pass org_id and user_id to organizations_remove_user."""
        organizations_client.remove_user(
            organization="T3JnYW5pemF0aW9uOjEyMzQ1",
            user_id="VXNlcjoxMjM0NQ==",
        )

        mock_api.organizations_remove_user.assert_called_once_with(
            org_id="T3JnYW5pemF0aW9uOjEyMzQ1",
            user_id="VXNlcjoxMjM0NQ==",
        )

    def test_remove_user_returns_none(
        self, organizations_client: OrganizationsClient, mock_api: Mock
    ) -> None:
        """remove_user() should return None on success (204 response)."""
        mock_api.organizations_remove_user.return_value = None

        result = organizations_client.remove_user(
            organization="T3JnYW5pemF0aW9uOjEyMzQ1",
            user_id="VXNlcjoxMjM0NQ==",
        )

        assert result is None

    def test_remove_user_emits_alpha_prerelease_warning(
        self,
        organizations_client: OrganizationsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        organizations_client.remove_user(
            organization="T3JnYW5pemF0aW9uOjEyMzQ1",
            user_id="VXNlcjoxMjM0NQ==",
        )

        assert any(
            "ALPHA" in record.message
            and "organizations.remove_user" in record.message
            for record in caplog.records
        )
