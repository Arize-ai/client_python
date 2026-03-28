"""Unit tests for src/arize/spaces/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.spaces.client import SpacesClient


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
        """list() should pass organization, limit, and cursor to spaces_list."""
        spaces_client.list(
            organization_id="org-123",
            limit=50,
            cursor="cursor-abc",
        )

        mock_api.spaces_list.assert_called_once_with(
            org_id="org-123",
            limit=50,
            cursor="cursor-abc",
        )

    def test_list_defaults(
        self, spaces_client: SpacesClient, mock_api: Mock
    ) -> None:
        """list() should default organization_id/cursor to None and limit to 100."""
        spaces_client.list()

        mock_api.spaces_list.assert_called_once_with(
            org_id=None,
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
