"""Unit tests for src/arize/datasets/client.py."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, Mock, create_autospec, patch

import pandas as pd
import pytest

from arize._generated.api_client import DatasetsApi
from arize.datasets.client import DatasetsClient


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock DatasetsApi instance."""
    return create_autospec(DatasetsApi, instance=True)


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

        mock_api.list_datasets.assert_called_once_with(
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

        mock_api.list_datasets.assert_called_once_with(
            space_id=None,
            space_name="my-space",
            name="my-dataset",
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_defaults(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """list() should default space/name/cursor to None and limit to 50."""
        datasets_client.list()

        mock_api.list_datasets.assert_called_once_with(
            space_id=None,
            space_name=None,
            name=None,
            limit=50,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from datasets_list."""
        expected = Mock()
        mock_api.list_datasets.return_value = expected

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


@pytest.mark.unit
class TestDatasetsClientListExamples:
    """Tests for DatasetsClient.list_examples() REST path."""

    # Base64-encoded dataset ID that bypasses name resolution
    DATASET_ID = "RGF0YXNldDoxMjM6YWJj"

    def test_list_examples_passes_cursor(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """list_examples() should forward cursor to the generated client."""
        datasets_client.list_examples(
            dataset=self.DATASET_ID,
            cursor="tok-abc",
        )

        mock_api.list_dataset_examples.assert_called_once_with(
            dataset_id=self.DATASET_ID,
            dataset_version_id=None,
            limit=50,
            cursor="tok-abc",
        )

    def test_list_examples_defaults_cursor_to_none(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """list_examples() should default cursor to None (first page)."""
        datasets_client.list_examples(dataset=self.DATASET_ID)

        mock_api.list_dataset_examples.assert_called_once_with(
            dataset_id=self.DATASET_ID,
            dataset_version_id=None,
            limit=50,
            cursor=None,
        )


@pytest.mark.unit
class TestDatasetsClientUpdateExamples:
    """Tests for DatasetsClient.update_examples()."""

    DATASET_ID = "RGF0YXNldDoxMjM6YWJj"

    def test_update_examples_builds_request_with_ids(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """update_examples() should build a UpdateDatasetExampleInput per example, keyed by id."""
        datasets_client.update_examples(
            dataset=self.DATASET_ID,
            examples=[
                {"id": "ex_1", "question": "2+2?", "answer": "4"},
                {"id": "ex_2", "question": "3+3?", "answer": "6"},
            ],
        )

        _, kwargs = mock_api.update_dataset_examples.call_args
        assert kwargs["dataset_id"] == self.DATASET_ID
        body = kwargs["update_dataset_examples_request"]
        assert [e.id for e in body.examples] == ["ex_1", "ex_2"]
        assert body.examples[0].additional_properties == {
            "question": "2+2?",
            "answer": "4",
        }

    def test_update_examples_defaults_new_version_to_none(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """Omitting new_version should update the targeted version in place."""
        datasets_client.update_examples(
            dataset=self.DATASET_ID,
            examples=[{"id": "ex_1", "answer": "4"}],
        )

        _, kwargs = mock_api.update_dataset_examples.call_args
        assert kwargs["update_dataset_examples_request"].new_version is None
        assert kwargs["dataset_version_id"] == ""

    def test_update_examples_passes_new_version(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """A non-empty new_version should be forwarded to create a new version."""
        datasets_client.update_examples(
            dataset=self.DATASET_ID,
            examples=[{"id": "ex_1", "answer": "4"}],
            new_version="v2",
        )

        _, kwargs = mock_api.update_dataset_examples.call_args
        assert kwargs["update_dataset_examples_request"].new_version == "v2"

    def test_update_examples_passes_dataset_version_id(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """An explicit dataset_version_id should target that version's in-place update."""
        datasets_client.update_examples(
            dataset=self.DATASET_ID,
            dataset_version_id="ver_1",
            examples=[{"id": "ex_1", "answer": "4"}],
        )

        _, kwargs = mock_api.update_dataset_examples.call_args
        assert kwargs["dataset_version_id"] == "ver_1"

    def test_update_examples_returns_api_response(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """update_examples() should propagate the return value from datasets_examples_update."""
        expected = Mock()
        mock_api.update_dataset_examples.return_value = expected

        result = datasets_client.update_examples(
            dataset=self.DATASET_ID,
            examples=[{"id": "ex_1", "answer": "4"}],
        )

        assert result is expected

    def test_update_examples_emits_beta_prerelease_warning(
        self,
        datasets_client: DatasetsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        datasets_client.update_examples(
            dataset=self.DATASET_ID,
            examples=[{"id": "ex_1", "answer": "4"}],
        )

        assert any(
            "BETA" in record.message
            and "datasets.update_examples" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestDatasetsClientDeleteExamples:
    """Tests for DatasetsClient.delete_examples()."""

    # Base64-encoded dataset ID that bypasses name resolution
    DATASET_ID = "RGF0YXNldDoxMjM6YWJj"

    def test_delete_examples_builds_request_and_forwards(
        self, datasets_client: DatasetsClient, mock_api: Mock
    ) -> None:
        """delete_examples() should build the request body and forward it."""
        result = datasets_client.delete_examples(
            dataset=self.DATASET_ID,
            dataset_version_id="ver-1",
            examples=["ex-1", "ex-2"],
        )

        mock_api.delete_dataset_examples.assert_called_once()
        call = mock_api.delete_dataset_examples.call_args
        assert call.kwargs["dataset_id"] == self.DATASET_ID
        body = call.kwargs["delete_dataset_examples_request"]
        assert body.dataset_version_id == "ver-1"
        assert body.example_ids == ["ex-1", "ex-2"]
        assert result is mock_api.delete_dataset_examples.return_value

    def test_delete_examples_rejects_empty_list(
        self, datasets_client: DatasetsClient
    ) -> None:
        """delete_examples() should reject an empty example list (min_length=1)."""
        with pytest.raises(Exception):
            datasets_client.delete_examples(
                dataset=self.DATASET_ID,
                dataset_version_id="ver-1",
                examples=[],
            )


@pytest.mark.unit
class TestDatasetsClientListExamplesCaching:
    """Tests for DatasetsClient.list_examples() caching behaviour."""

    def _make_client(
        self, mock_sdk_config: Mock, enable_caching: bool
    ) -> DatasetsClient:
        mock_sdk_config.enable_caching = enable_caching
        with patch(
            "arize._generated.api_client.DatasetsApi", return_value=Mock()
        ):
            return DatasetsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )

    def test_cache_write_skipped_when_caching_disabled(
        self, mock_sdk_config: Mock
    ) -> None:
        """list_examples(all=True) must not write to cache when enable_caching=False."""
        client = self._make_client(mock_sdk_config, enable_caching=False)

        dataset_obj = Mock()
        dataset_obj.updated_at = "2024-01-01T00:00:00Z"
        dataset_obj.space_id = "space-123"

        empty_df = pd.DataFrame(columns=["id", "input", "output"])

        with (
            patch.object(client, "get", return_value=dataset_obj),
            patch(
                "arize.datasets.client.load_cached_resource", return_value=None
            ),
            patch("arize.datasets.client.cache_resource") as mock_cache_write,
            patch("arize.datasets.client.ArizeFlightClient") as mock_flight_cls,
        ):
            mock_flight_instance = MagicMock()
            mock_flight_instance.__enter__ = Mock(
                return_value=mock_flight_instance
            )
            mock_flight_instance.__exit__ = Mock(return_value=False)
            mock_flight_instance.get_dataset_examples.return_value = empty_df
            mock_flight_cls.return_value = mock_flight_instance

            # Use a base64-encoded ID so _find_dataset_id treats it as a
            # direct resource ID and skips the name-lookup API call.
            client.list_examples(dataset="RGF0YXNldDoxMjM6YWJj", all=True)

        mock_cache_write.assert_not_called()

    def test_cache_write_called_when_caching_enabled(
        self, mock_sdk_config: Mock
    ) -> None:
        """list_examples(all=True) must write to cache when enable_caching=True."""
        client = self._make_client(mock_sdk_config, enable_caching=True)

        dataset_obj = Mock()
        dataset_obj.updated_at = "2024-01-01T00:00:00Z"
        dataset_obj.space_id = "space-123"

        empty_df = pd.DataFrame(columns=["id", "input", "output"])

        with (
            patch.object(client, "get", return_value=dataset_obj),
            patch(
                "arize.datasets.client.load_cached_resource", return_value=None
            ),
            patch("arize.datasets.client.cache_resource") as mock_cache_write,
            patch("arize.datasets.client.ArizeFlightClient") as mock_flight_cls,
        ):
            mock_flight_instance = MagicMock()
            mock_flight_instance.__enter__ = Mock(
                return_value=mock_flight_instance
            )
            mock_flight_instance.__exit__ = Mock(return_value=False)
            mock_flight_instance.get_dataset_examples.return_value = empty_df
            mock_flight_cls.return_value = mock_flight_instance

            # Use a base64-encoded ID so _find_dataset_id treats it as a
            # direct resource ID and skips the name-lookup API call.
            client.list_examples(dataset="RGF0YXNldDoxMjM6YWJj", all=True)

        mock_cache_write.assert_called_once()
