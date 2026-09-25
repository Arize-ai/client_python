"""Unit tests for src/arize/datasets/client.py."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, create_autospec, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arize._generated.api_client import DatasetsApi
from arize.datasets import errors as err
from arize.datasets.client import DatasetsClient
from arize.datasets.upload import flight_schema


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

    @pytest.mark.parametrize("from_cache", [False, True])
    def test_all_normalizes_numpy_values(
        self,
        mock_sdk_config: Mock,
        from_cache: bool,
    ) -> None:
        """Flight and cached examples must contain JSON-compatible values."""
        client = self._make_client(mock_sdk_config, enable_caching=from_cache)
        dataset_obj = Mock(
            updated_at="2024-01-01T00:00:00Z",
            space_id="space-123",
        )
        now = datetime.now(tz=timezone.utc)
        timestamps = (
            pa.table(
                {
                    "timestamps": pa.array(
                        [[datetime(2024, 1, 1), None]],
                        type=pa.list_(pa.timestamp("ns")),
                    )
                }
            )
            .to_pandas()["timestamps"]
            .iloc[0]
        )
        dataset_df = pd.DataFrame(
            {
                "id": ["example-1"],
                "created_at": [now],
                "updated_at": [now],
                "expected_tool_names": [np.array(["search", "lookup"])],
                "timestamps": [timestamps],
                "output": [
                    {
                        "expected_tool_calls": np.array(
                            [{"name": "search"}], dtype=object
                        ),
                        "expected_tool_count": np.int64(1),
                    }
                ],
            }
        )

        with (
            patch.object(client, "get", return_value=dataset_obj),
            patch(
                "arize.datasets.client.load_cached_resource",
                return_value=dataset_df if from_cache else None,
            ),
            patch("arize.datasets.client.cache_resource"),
            patch("arize.datasets.client.ArizeFlightClient") as flight_cls,
        ):
            flight_client = MagicMock()
            flight_client.__enter__ = Mock(return_value=flight_client)
            flight_client.__exit__ = Mock(return_value=False)
            flight_client.get_dataset_examples.return_value = dataset_df
            flight_cls.return_value = flight_client

            response = client.list_examples(
                dataset="RGF0YXNldDoxMjM6YWJj", all=True
            )

        example = response.examples[0]
        assert example.additional_properties["expected_tool_names"] == [
            "search",
            "lookup",
        ]
        assert example.additional_properties["output"] == {
            "expected_tool_calls": [{"name": "search"}],
            "expected_tool_count": 1,
        }
        assert example.additional_properties["timestamps"] == [
            "2024-01-01T00:00:00.000000000",
            None,
        ]
        model_dump = response.model_dump(mode="json")
        assert model_dump["examples"][0]["additional_properties"][
            "expected_tool_names"
        ] == ["search", "lookup"]
        serialized = json.loads(response.to_json())
        assert serialized["examples"][0]["expected_tool_names"] == [
            "search",
            "lookup",
        ]


DATASET_ID = "RGF0YXNldDoxMjM6YWJj"


@pytest.mark.unit
class TestDatasetsClientCreate:
    """Tests for DatasetsClient.create()."""

    @pytest.fixture(autouse=True)
    def _configure(self, mock_sdk_config: Mock) -> None:
        mock_sdk_config.pyarrow_max_chunksize = 10
        mock_sdk_config.max_http_payload_size_mb = 8

    @pytest.fixture
    def flight_client(self) -> MagicMock:
        client = MagicMock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.written = []

        def create_dataset(**kwargs: object) -> str:
            client.written.extend(kwargs["reader"])  # type: ignore[attr-defined]
            return DATASET_ID

        client.create_dataset.side_effect = create_dataset
        return client

    @pytest.fixture
    def parquet_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "examples.parquet"
        pq.write_table(
            pa.table({"id": ["a", "b", "c"], "query": ["q1", "q2", "q3"]}),
            path,
        )
        return path

    def _create(
        self,
        datasets_client: DatasetsClient,
        flight_client: MagicMock,
        examples: object,
        **kwargs: object,
    ) -> tuple[Mock, Mock, Mock]:
        with (
            patch(
                "arize.datasets.client._find_space_id", return_value="space-1"
            ) as find_space,
            patch(
                "arize.datasets.client.ArizeFlightClient",
                return_value=flight_client,
            ) as flight_cls,
            patch.object(DatasetsClient, "get", return_value=Mock()) as get,
        ):
            datasets_client.create(
                name="ds",
                space="space",
                examples=examples,  # type: ignore[arg-type]
                **kwargs,  # type: ignore[arg-type]
            )
        return find_space, flight_cls, get

    def test_small_list_uses_rest(
        self,
        datasets_client: DatasetsClient,
        mock_api: Mock,
        flight_client: MagicMock,
    ) -> None:
        _, flight_cls, _ = self._create(
            datasets_client, flight_client, [{"query": "q"}]
        )
        mock_api.create_dataset.assert_called_once()
        flight_cls.assert_not_called()

    def test_large_dataframe_streams_table_batches(
        self,
        datasets_client: DatasetsClient,
        mock_api: Mock,
        flight_client: MagicMock,
    ) -> None:
        df = pd.DataFrame({"query": [f"q{i}" for i in range(25)]})
        with patch(
            "arize.datasets.client.get_payload_size_mb", return_value=100.0
        ):
            _, _, get = self._create(datasets_client, flight_client, df)
        mock_api.create_dataset.assert_not_called()
        kwargs = flight_client.create_dataset.call_args.kwargs
        batches = flight_client.written
        assert [b.num_rows for b in batches] == [10, 10, 5]
        assert all(b.schema.equals(kwargs["reader"].schema) for b in batches)
        get.assert_called_once_with(dataset=DATASET_ID)

    def test_empty_list_raises(
        self, datasets_client: DatasetsClient, flight_client: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="empty dataset"):
            self._create(datasets_client, flight_client, [])

    @pytest.mark.parametrize(
        "make_input",
        [
            str,
            Path,
            lambda p: [p],
            lambda p: (str(p),),
            lambda p: p.parent,
        ],
        ids=["str", "path", "list", "tuple", "directory"],
    )
    def test_file_input_streams_via_flight(
        self,
        datasets_client: DatasetsClient,
        mock_api: Mock,
        flight_client: MagicMock,
        parquet_file: Path,
        make_input: object,
    ) -> None:
        examples = make_input(parquet_file)  # type: ignore[operator]
        _, flight_cls, get = self._create(
            datasets_client, flight_client, examples
        )
        mock_api.create_dataset.assert_not_called()
        flight_cls.assert_called_once()
        kwargs = flight_client.create_dataset.call_args.kwargs
        assert kwargs["reader"].schema.equals(
            flight_schema(pq.read_schema(parquet_file))
        )
        table = pa.Table.from_batches(flight_client.written)
        assert table.column("id").to_pylist() == ["a", "b", "c"]
        assert table.column("query").to_pylist() == ["q1", "q2", "q3"]
        get.assert_called_once_with(dataset=DATASET_ID)

    def test_arrow_file_input(
        self,
        datasets_client: DatasetsClient,
        flight_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "examples.arrow"
        table = pa.table({"query": ["q1", "q2"]})
        with (
            pa.OSFile(str(path), "wb") as sink,
            pa.ipc.new_file(sink, table.schema) as writer,
        ):
            writer.write_table(table)
        self._create(datasets_client, flight_client, path)
        out = pa.Table.from_batches(flight_client.written)
        assert out.column("query").to_pylist() == ["q1", "q2"]
        assert out.schema.names == ["query", "id", "created_at", "updated_at"]

    def test_force_http_with_path_raises_before_lookup(
        self,
        datasets_client: DatasetsClient,
        flight_client: MagicMock,
        parquet_file: Path,
    ) -> None:
        with (
            patch("arize.datasets.client._find_space_id") as find_space,
            pytest.raises(ValueError, match="force_http"),
        ):
            datasets_client.create(
                name="ds", space="space", examples=parquet_file, force_http=True
            )
        find_space.assert_not_called()

    def test_mixed_list_raises_type_error(
        self,
        datasets_client: DatasetsClient,
        flight_client: MagicMock,
        parquet_file: Path,
    ) -> None:
        with pytest.raises(TypeError, match="only dicts or only file paths"):
            self._create(
                datasets_client,
                flight_client,
                [{"query": "q"}, str(parquet_file)],
            )

    @pytest.mark.parametrize(
        ("table", "error"),
        [
            (
                pa.table({"query": pa.array([], pa.string())}),
                err.EmptyDatasetError,
            ),
            (pa.table({"raw": [b"x"]}), err.BinaryColumnError),
            (pa.table({"id": ["a", "a"]}), err.IDColumnUniqueConstraintError),
        ],
        ids=["empty", "binary", "duplicate-ids"],
    )
    def test_pre_stream_checks_fail_before_flight(
        self,
        datasets_client: DatasetsClient,
        flight_client: MagicMock,
        tmp_path: Path,
        table: pa.Table,
        error: type[Exception],
    ) -> None:
        path = tmp_path / "bad.parquet"
        pq.write_table(table, path)
        with (
            patch(
                "arize.datasets.client._find_space_id", return_value="space-1"
            ),
            patch("arize.datasets.client.ArizeFlightClient") as flight_cls,
            pytest.raises(error),
        ):
            datasets_client.create(name="ds", space="space", examples=path)
        flight_cls.assert_not_called()

    def test_logs_streaming_summary_without_sizing_payload(
        self,
        datasets_client: DatasetsClient,
        flight_client: MagicMock,
        parquet_file: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with (
            patch(
                "arize.datasets.client.get_payload_size_mb",
                side_effect=AssertionError("must not size file input"),
            ),
            caplog.at_level(logging.INFO, logger="arize.datasets.client"),
        ):
            self._create(datasets_client, flight_client, parquet_file)
        assert "Streaming 3 examples from 1 file(s)" in caplog.text
