"""Unit tests for src/arize/datasets/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.datasets.client import DatasetsClient


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock DatasetsApi instance."""
    return Mock()


@pytest.fixture
def datasets_client(mock_sdk_config: Mock, mock_api: Mock) -> DatasetsClient:
    """Provide a DatasetsClient with mocked internals."""
    with patch(
        "arize._generated.api_client.DatasetsApi", return_value=mock_api
    ):
        return DatasetsClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestDatasetsClientInit:
    """Tests for DatasetsClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.DatasetsApi", return_value=mock_api
        ):
            client = DatasetsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_datasets_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to DatasetsApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.DatasetsApi"
        ) as mock_datasets_api_cls:
            DatasetsClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_datasets_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestDatasetsClientList:
    """Tests for DatasetsClient.list()."""

    def test_list_with_space_id(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """list() should resolve a base64 resource ID space value to space_id."""
        datasets_client.list(
            name="my-dataset",
            space="U3BhY2U6OTA1MDoxSmtS",
            limit=25,
            cursor="cursor-xyz",
        )

        mock_api.datasets_list.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS",
            space_name=None,
            name="my-dataset",
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_with_space_name(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """list() should resolve a non-prefixed space value to space_name."""
        datasets_client.list(
            name="my-dataset",
            space="my-space",
            limit=25,
            cursor="cursor-xyz",
        )

        mock_api.datasets_list.assert_called_once_with(
            space_id=None,
            space_name="my-space",
            name="my-dataset",
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_defaults(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """list() should default space/name/cursor to None and limit to 100."""
        datasets_client.list()

        mock_api.datasets_list.assert_called_once_with(
            space_id=None,
            space_name=None,
            name=None,
            limit=100,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from datasets_list."""
        expected = Mock()
        mock_api.datasets_list.return_value = expected

        result = datasets_client.list()

        assert result is expected

    def test_list_emits_beta_prerelease_warning(
        self,
        datasets_client: DatasetsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        datasets_client.list()

        assert any(
            "BETA" in record.message and "datasets.list" in record.message
            for record in caplog.records
        )
