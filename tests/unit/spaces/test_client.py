"""Unit tests for src/arize/spaces/client.py."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from arize.spaces.client import SpacesClient
from arize.spaces.types import (
    PredefinedSpaceRole,
    SpaceMembership,
    UserSpaceRole,
)


@pytest.fixture(autouse=True)
def _stub_from_generated() -> Generator[None, None, None]:
    """Stub model_validate on domain types so tests that don't explicitly
    test conversion don't fail when the client calls it on a Mock API response.
    """
    with patch.object(SpaceMembership, "model_validate", return_value=Mock()):
        yield


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock SpacesApi instance."""
    return Mock()


@pytest.fixture
def spaces_client(mock_sdk_config: Mock, mock_api: Mock) -> SpacesClient:
    """Provide a SpacesClient with mocked internals."""
    with patch("arize._generated.api_client.SpacesApi", return_value=mock_api):
        return SpacesClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestSpacesClientInit:
    """Tests for SpacesClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.SpacesApi", return_value=mock_api
        ):
            client = SpacesClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_spaces_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to SpacesApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.SpacesApi"
        ) as mock_spaces_api_cls:
            SpacesClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_spaces_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestSpacesClientList:
    """Tests for SpacesClient.list()."""

    def test_list_calls_api_with_all_params(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """list() should pass organization, name, limit, and cursor to spaces_list."""
        spaces_client.list(
            organization_id="org-123",
            name="prod-space",
            limit=50,
            cursor="cursor-abc",
        )

        mock_api.spaces_list.assert_called_once_with(
            org_id="org-123",
            name="prod-space",
            limit=50,
            cursor="cursor-abc",
        )

    def test_list_defaults(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """list() should default organization_id/name/cursor to None and limit to 100."""
        spaces_client.list()

        mock_api.spaces_list.assert_called_once_with(
            org_id=None,
            name=None,
            limit=100,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from spaces_list."""
        expected = Mock()
        mock_api.spaces_list.return_value = expected

        result = spaces_client.list()

        assert result is expected

    def test_list_emits_beta_prerelease_warning(
        self,
        spaces_client: SpacesClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        spaces_client.list()

        assert any(
            "BETA" in record.message and "spaces.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestSpacesClientGet:
    """Tests for SpacesClient.get()."""

    def test_get_calls_api_with_space_id(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """get() should resolve space and pass space_id to spaces_get."""
        spaces_client.get(space="U3BhY2U6OTA1MDoxSmtS")

        mock_api.spaces_get.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS"
        )

    def test_get_returns_api_response(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """get() should propagate the return value from spaces_get."""
        expected = Mock()
        mock_api.spaces_get.return_value = expected

        result = spaces_client.get(space="U3BhY2U6OTA1MDoxSmtS")

        assert result is expected


@pytest.mark.unit
class TestSpacesClientCreate:
    """Tests for SpacesClient.create()."""

    def test_create_builds_request_and_calls_api(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """create() should build SpacesCreateRequest and pass it to spaces_create."""
        with patch(
            "arize._generated.api_client.SpacesCreateRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            spaces_client.create(
                name="my-space",
                organization_id="org-123",
                description="my description",
            )

        mock_request_cls.assert_called_once_with(
            name="my-space",
            organization_id="org-123",
            description="my description",
        )
        mock_api.spaces_create.assert_called_once_with(
            spaces_create_request=mock_body
        )

    def test_create_returns_api_response(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from spaces_create."""
        expected = Mock()
        mock_api.spaces_create.return_value = expected

        with patch("arize._generated.api_client.SpacesCreateRequest"):
            result = spaces_client.create(
                name="my-space",
                organization_id="org-123",
            )

        assert result is expected


@pytest.mark.unit
class TestSpacesClientDelete:
    """Tests for SpacesClient.delete()."""

    def test_delete_calls_api_with_space_id(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """delete() should resolve space and pass space_id to spaces_delete."""
        spaces_client.delete(space="U3BhY2U6OTA1MDoxSmtS")

        mock_api.spaces_delete.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS"
        )

    def test_delete_returns_none(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """delete() should return None on success (204 response)."""
        mock_api.spaces_delete.return_value = None

        result = spaces_client.delete(space="U3BhY2U6OTA1MDoxSmtS")

        assert result is None

    def test_delete_emits_alpha_prerelease_warning(
        self,
        spaces_client: SpacesClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        spaces_client.delete(space="U3BhY2U6OTA1MDoxSmtS")

        assert any(
            "ALPHA" in record.message and "spaces.delete" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestSpacesClientUpdate:
    """Tests for SpacesClient.update()."""

    def test_update_raises_when_no_fields_provided(
        self, spaces_client: SpacesClient
    ) -> None:
        """update() should raise if neither name nor description is provided."""
        with pytest.raises(
            ValueError,
            match="At least one of 'name' or 'description' must be provided",
        ):
            spaces_client.update(space="U3BhY2U6OTA1MDoxSmtS")

    def test_update_builds_request_and_calls_api(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """update() should build SpacesUpdateRequest and pass it to spaces_update."""
        with patch(
            "arize._generated.api_client.SpacesUpdateRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            spaces_client.update(
                space="U3BhY2U6OTA1MDoxSmtS",
                name="updated-space",
                description="updated description",
            )

        mock_request_cls.assert_called_once_with(
            name="updated-space",
            description="updated description",
        )
        mock_api.spaces_update.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS",
            spaces_update_request=mock_body,
        )

    def test_update_returns_api_response(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """update() should propagate the return value from spaces_update."""
        expected = Mock()
        mock_api.spaces_update.return_value = expected

        with patch("arize._generated.api_client.SpacesUpdateRequest"):
            result = spaces_client.update(
                space="U3BhY2U6OTA1MDoxSmtS",
                name="updated-space",
            )

        assert result is expected


@pytest.mark.unit
class TestSpacesClientAddUser:
    """Tests for SpacesClient.add_user()."""

    def test_add_user_with_predefined_role_calls_api(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """add_user() with PredefinedSpaceRole should wrap it in SpaceRoleAssignment."""
        role = PredefinedSpaceRole(name=UserSpaceRole.MEMBER)
        with (
            patch(
                "arize._generated.api_client.SpaceMembershipInput"
            ) as mock_input_cls,
            patch(
                "arize._generated.api_client.SpaceRoleAssignment"
            ) as mock_role_cls,
            patch(
                "arize._generated.api_client.PredefinedRoleAssignment"
            ) as mock_pred_cls,
        ):
            mock_body = Mock()
            mock_input_cls.return_value = mock_body
            mock_role = Mock()
            mock_role_cls.return_value = mock_role
            mock_gen_role = Mock()
            mock_pred_cls.return_value = mock_gen_role

            spaces_client.add_user(
                space="U3BhY2U6OTA1MDoxSmtS",
                user_id="VXNlcjoxMjM0NQ==",
                role=role,
            )

        mock_pred_cls.assert_called_once()
        mock_role_cls.assert_called_once_with(mock_gen_role)
        mock_input_cls.assert_called_once_with(
            user_id="VXNlcjoxMjM0NQ==",
            role=mock_role,
        )
        mock_api.spaces_add_user.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS",
            space_membership_input=mock_body,
        )

    def test_add_user_returns_domain_membership(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """add_user() should convert the raw API response to a domain SpaceMembership."""
        raw = Mock()
        mock_api.spaces_add_user.return_value = raw
        domain = Mock()

        with (
            patch("arize._generated.api_client.SpaceMembershipInput"),
            patch("arize._generated.api_client.SpaceRoleAssignment"),
            patch.object(
                SpaceMembership, "model_validate", return_value=domain
            ) as mock_conv,
        ):
            result = spaces_client.add_user(
                space="U3BhY2U6OTA1MDoxSmtS",
                user_id="VXNlcjoxMjM0NQ==",
                role=PredefinedSpaceRole(name=UserSpaceRole.MEMBER),
            )

        mock_conv.assert_called_once_with(raw, from_attributes=True)
        assert result is domain

    def test_add_user_emits_alpha_prerelease_warning(
        self,
        spaces_client: SpacesClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch("arize._generated.api_client.SpaceMembershipInput"),
            patch("arize._generated.api_client.SpaceRoleAssignment"),
        ):
            spaces_client.add_user(
                space="U3BhY2U6OTA1MDoxSmtS",
                user_id="VXNlcjoxMjM0NQ==",
                role=PredefinedSpaceRole(name=UserSpaceRole.MEMBER),
            )

        assert any(
            "ALPHA" in record.message and "spaces.add_user" in record.message
            for record in caplog.records
        )
