"""Unit tests for arize._exporter.client module."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import Mock, call, patch

import pandas as pd
import pyarrow as pa
import pytest

from arize._exporter.client import (
    ArizeExportClient,
    _get_pb_similarity_search_params,
)
from arize.ml.types import (
    Environments,
    SimilarityReference,
    SimilaritySearchParams,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
class TestArizeExportClientInit:
    """Test ArizeExportClient initialization."""

    def test_init_with_flight_client(self, mock_flight_client: Mock) -> None:
        """Test initialization with FlightClient."""
        client = ArizeExportClient(flight_client=mock_flight_client)
        assert client.flight_client is mock_flight_client

    def test_frozen_dataclass_immutable(
        self, export_client: ArizeExportClient
    ) -> None:
        """Test that ArizeExportClient is frozen."""
        with pytest.raises(AttributeError):
            export_client.flight_client = Mock()  # type: ignore[misc]


@pytest.mark.unit
class TestGetStreamReader:
    """Test _get_stream_reader method."""

    @pytest.mark.parametrize(
        "param_name,invalid_value,error_pattern",
        [
            ("space_id", 123, r"space_id.*must be a str"),
            ("model_id", 456, r"model_id.*must be a str"),
            ("environment", "production", r"environment.*must be"),
            ("start_time", "2024-01-01", r"start_time.*must be a datetime"),
            ("end_time", "2024-01-02", r"end_time.*must be a datetime"),
        ],
    )
    def test_validates_parameter_type(
        self,
        export_client: ArizeExportClient,
        valid_export_params: dict,
        param_name: str,
        invalid_value: object,
        error_pattern: str,
    ) -> None:
        """Test that parameter types are validated."""
        with pytest.raises(TypeError, match=error_pattern):
            params = {
                k: v for k, v in valid_export_params.items() if k != param_name
            }
            export_client._get_stream_reader(
                **{param_name: invalid_value, **params}
            )  # type: ignore[arg-type]

    def test_validates_start_before_end(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test that start_time must be before end_time."""
        params = valid_export_params.copy()
        params["start_time"] = datetime(2024, 1, 2)
        params["end_time"] = datetime(2024, 1, 1)
        with pytest.raises(
            ValueError, match="start_time must be before end_time"
        ):
            export_client._get_stream_reader(**params)

    def test_validates_stream_chunk_size_type(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test that stream_chunk_size type is validated."""
        with pytest.raises(
            TypeError, match=r"stream_chunk_size.*must be a int"
        ):
            export_client._get_stream_reader(
                **valid_export_params, stream_chunk_size="100"
            )  # type: ignore[arg-type]

    def test_returns_stream_reader_and_record_count(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test successful return of stream reader and record count."""
        # Mock FlightInfo
        mock_flight_info = Mock()
        mock_flight_info.total_records = 100
        mock_endpoint = Mock()
        mock_endpoint.ticket = Mock()
        mock_flight_info.endpoints = [mock_endpoint]
        export_client.flight_client.get_flight_info.return_value = (
            mock_flight_info
        )

        # Mock do_get
        mock_reader = Mock()
        export_client.flight_client.do_get.return_value = mock_reader

        reader, count = export_client._get_stream_reader(**valid_export_params)

        assert reader is mock_reader
        assert count == 100
        assert export_client.flight_client.get_flight_info.called
        assert export_client.flight_client.do_get.called

    def test_includes_similarity_search_params_when_provided(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test that similarity search params are included when provided."""
        similarity_params = SimilaritySearchParams(
            search_column_name="embedding",
            threshold=0.8,
            references=[
                SimilarityReference(
                    prediction_id="pred123",
                    reference_column_name="ref_embedding",
                    prediction_timestamp=datetime(2024, 1, 1),
                )
            ],
        )

        # Mock FlightInfo
        mock_flight_info = Mock()
        mock_flight_info.total_records = 10
        mock_endpoint = Mock()
        mock_endpoint.ticket = Mock()
        mock_flight_info.endpoints = [mock_endpoint]
        export_client.flight_client.get_flight_info.return_value = (
            mock_flight_info
        )
        export_client.flight_client.do_get.return_value = Mock()

        export_client._get_stream_reader(
            **valid_export_params,
            similarity_search_params=similarity_params,
        )

        assert export_client.flight_client.get_flight_info.called

    def test_includes_columns_filter_when_provided(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test that column filter is included when provided."""
        columns = ["col1", "col2", "col3"]

        # Mock FlightInfo
        mock_flight_info = Mock()
        mock_flight_info.total_records = 10
        mock_endpoint = Mock()
        mock_endpoint.ticket = Mock()
        mock_flight_info.endpoints = [mock_endpoint]
        export_client.flight_client.get_flight_info.return_value = (
            mock_flight_info
        )
        export_client.flight_client.do_get.return_value = Mock()

        export_client._get_stream_reader(**valid_export_params, columns=columns)

        assert export_client.flight_client.get_flight_info.called

    def test_flight_info_error_raises_runtime_error(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test that FlightInfo errors raise RuntimeError."""
        export_client.flight_client.get_flight_info.side_effect = Exception(
            "Connection error"
        )

        with pytest.raises(RuntimeError, match="Error getting flight info"):
            export_client._get_stream_reader(**valid_export_params)

    def test_zero_records_returns_empty_reader(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test that zero records returns None reader."""
        # Mock FlightInfo with zero records
        mock_flight_info = Mock()
        mock_flight_info.total_records = 0
        export_client.flight_client.get_flight_info.return_value = (
            mock_flight_info
        )

        reader, count = export_client._get_stream_reader(**valid_export_params)

        assert reader is None
        assert count == 0


@pytest.mark.unit
class TestExportToDf:
    """Test export_to_df method."""

    def test_exports_single_chunk(
        self,
        export_client: ArizeExportClient,
        valid_export_params: dict,
        sample_arrow_table: pa.Table,
    ) -> None:
        """Test exporting a single chunk."""
        # Mock _get_stream_reader
        mock_reader = Mock()
        mock_batch = Mock()
        mock_batch.data = sample_arrow_table
        mock_reader.read_chunk.side_effect = [mock_batch, StopIteration()]

        # Using `with patch()` context manager instead of `@patch` decorator because:
        # 1. ArizeExportClient is a frozen dataclass, so patch.object() would fail
        # 2. We set up mock objects (mock_reader, mock_batch) before patching
        # 3. Explicit scope - the indentation clearly shows where the mock is active
        # Alternative: Use `@patch` decorator when mocking applies to entire test function
        with patch(
            "arize._exporter.client.ArizeExportClient._get_stream_reader",
            return_value=(mock_reader, 3),
        ):
            result = export_client.export_to_df(**valid_export_params)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "time" in result.columns
        assert "value" in result.columns

    def test_exports_multiple_chunks(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test exporting multiple chunks."""
        # Create multiple chunks
        chunk1 = pa.table({"time": [1000, 2000], "value": [1.0, 2.0]})
        chunk2 = pa.table({"time": [3000, 4000], "value": [3.0, 4.0]})

        mock_reader = Mock()
        mock_batch1 = Mock()
        mock_batch1.data = chunk1
        mock_batch2 = Mock()
        mock_batch2.data = chunk2
        mock_reader.read_chunk.side_effect = [
            mock_batch1,
            mock_batch2,
            StopIteration(),
        ]

        with patch(
            "arize._exporter.client.ArizeExportClient._get_stream_reader",
            return_value=(mock_reader, 4),
        ):
            result = export_client.export_to_df(**valid_export_params)

        assert len(result) == 4

    def test_removes_null_columns(
        self,
        export_client: ArizeExportClient,
        valid_export_params: dict,
        sample_arrow_table: pa.Table,
    ) -> None:
        """Test that null columns are removed."""
        mock_reader = Mock()
        mock_batch = Mock()
        mock_batch.data = sample_arrow_table
        mock_reader.read_chunk.side_effect = [mock_batch, StopIteration()]

        with patch(
            "arize._exporter.client.ArizeExportClient._get_stream_reader",
            return_value=(mock_reader, 3),
        ):
            result = export_client.export_to_df(**valid_export_params)

        assert "null_col" not in result.columns

    def test_sorts_by_time_column(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test that result is sorted by time column."""
        # Create unsorted data
        unsorted_table = pa.table(
            {
                "time": [3000, 1000, 2000],
                "value": [3.0, 1.0, 2.0],
            }
        )

        mock_reader = Mock()
        mock_batch = Mock()
        mock_batch.data = unsorted_table
        mock_reader.read_chunk.side_effect = [mock_batch, StopIteration()]

        with patch(
            "arize._exporter.client.ArizeExportClient._get_stream_reader",
            return_value=(mock_reader, 3),
        ):
            result = export_client.export_to_df(**valid_export_params)

        assert result["time"].tolist() == [1000, 2000, 3000]

    def test_resets_index(
        self,
        export_client: ArizeExportClient,
        valid_export_params: dict,
        sample_arrow_table: pa.Table,
    ) -> None:
        """Test that index is reset."""
        mock_reader = Mock()
        mock_batch = Mock()
        mock_batch.data = sample_arrow_table
        mock_reader.read_chunk.side_effect = [mock_batch, StopIteration()]

        with patch(
            "arize._exporter.client.ArizeExportClient._get_stream_reader",
            return_value=(mock_reader, 3),
        ):
            result = export_client.export_to_df(**valid_export_params)

        assert result.index.tolist() == [0, 1, 2]

    @pytest.mark.parametrize(
        "environment,table_fixture,expected_len",
        [
            (Environments.PRODUCTION, "sample_arrow_table", 3),
            (Environments.TRACING, "sample_arrow_table_tracing", None),
        ],
    )
    def test_export_with_different_environments(
        self,
        export_client: ArizeExportClient,
        environment: Environments,
        table_fixture: str,
        expected_len: int | None,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test that export handles different environments correctly."""
        table = request.getfixturevalue(table_fixture)
        mock_reader = Mock()
        mock_batch = Mock()
        mock_batch.data = table
        mock_reader.read_chunk.side_effect = [mock_batch, StopIteration()]

        with patch(
            "arize._exporter.client.ArizeExportClient._get_stream_reader",
            return_value=(mock_reader, 3),
        ):
            result = export_client.export_to_df(
                space_id="space123",
                model_id="model456",
                environment=environment,
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 1, 2),
            )

        assert isinstance(result, pd.DataFrame)
        if expected_len is not None:
            assert len(result) == expected_len

    def test_empty_result_returns_empty_dataframe(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test that empty result returns empty DataFrame."""
        with patch(
            "arize._exporter.client.ArizeExportClient._get_stream_reader",
            return_value=(None, 0),
        ):
            result = export_client.export_to_df(**valid_export_params)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_handles_missing_time_column(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test handling when time column is missing."""
        table = pa.table({"value": [1.0, 2.0, 3.0]})

        mock_reader = Mock()
        mock_batch = Mock()
        mock_batch.data = table
        mock_reader.read_chunk.side_effect = [mock_batch, StopIteration()]

        # Should raise KeyError because time column is expected
        with (
            patch(
                "arize._exporter.client.ArizeExportClient._get_stream_reader",
                return_value=(mock_reader, 3),
            ),
            pytest.raises(KeyError, match="time"),
        ):
            export_client.export_to_df(**valid_export_params)

    def test_progress_bar_updates_correctly(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test that progress bar updates with correct counts."""
        chunk1 = pa.table({"time": [1000, 2000], "value": [1.0, 2.0]})
        chunk2 = pa.table({"time": [3000], "value": [3.0]})

        mock_reader = Mock()
        mock_batch1 = Mock()
        mock_batch1.data = chunk1
        mock_batch2 = Mock()
        mock_batch2.data = chunk2
        mock_reader.read_chunk.side_effect = [
            mock_batch1,
            mock_batch2,
            StopIteration(),
        ]

        with (
            patch(
                "arize._exporter.client.ArizeExportClient._get_stream_reader",
                return_value=(mock_reader, 3),
            ),
            patch("arize._exporter.client.tqdm") as mock_tqdm,
        ):
            mock_progress_bar = Mock()
            mock_tqdm.return_value = mock_progress_bar

            export_client.export_to_df(**valid_export_params)

            # Should be called twice with chunk sizes
            assert mock_progress_bar.update.call_count == 2
            mock_progress_bar.update.assert_has_calls([call(2), call(1)])

    def test_concatenates_chunks_correctly(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test that multiple chunks are concatenated correctly."""
        chunk1 = pa.table({"time": [1000], "value": [1.0], "label": ["a"]})
        chunk2 = pa.table({"time": [2000], "value": [2.0], "label": ["b"]})
        chunk3 = pa.table({"time": [3000], "value": [3.0], "label": ["c"]})

        mock_reader = Mock()
        mock_batch1, mock_batch2, mock_batch3 = Mock(), Mock(), Mock()
        mock_batch1.data = chunk1
        mock_batch2.data = chunk2
        mock_batch3.data = chunk3
        mock_reader.read_chunk.side_effect = [
            mock_batch1,
            mock_batch2,
            mock_batch3,
            StopIteration(),
        ]

        with patch(
            "arize._exporter.client.ArizeExportClient._get_stream_reader",
            return_value=(mock_reader, 3),
        ):
            result = export_client.export_to_df(**valid_export_params)

        assert len(result) == 3
        assert result["label"].tolist() == ["a", "b", "c"]


@pytest.mark.unit
class TestExportToParquet:
    """Test export_to_parquet method."""

    def test_writes_single_chunk_to_parquet(
        self,
        export_client: ArizeExportClient,
        valid_export_params: dict,
        sample_arrow_table: pa.Table,
        tmp_path: Path,
    ) -> None:
        """Test writing a single chunk to parquet."""
        output_path = str(tmp_path / "output.parquet")

        mock_reader = Mock()
        mock_reader.schema = sample_arrow_table.schema
        mock_batch = Mock()
        mock_batch.data = sample_arrow_table.to_batches()[0]
        mock_reader.read_chunk.side_effect = [mock_batch, StopIteration()]

        with patch(
            "arize._exporter.client.ArizeExportClient._get_stream_reader",
            return_value=(mock_reader, 3),
        ):
            export_client.export_to_parquet(
                path=output_path, **valid_export_params
            )

        # Verify file was created
        assert tmp_path.joinpath("output.parquet").exists()

    def test_writes_multiple_chunks_to_parquet(
        self,
        export_client: ArizeExportClient,
        valid_export_params: dict,
        tmp_path: Path,
    ) -> None:
        """Test writing multiple chunks to parquet."""
        output_path = str(tmp_path / "output.parquet")

        schema = pa.schema([("time", pa.int64()), ("value", pa.float64())])
        chunk1 = pa.record_batch([[1000, 2000], [1.0, 2.0]], schema=schema)
        chunk2 = pa.record_batch([[3000, 4000], [3.0, 4.0]], schema=schema)

        mock_reader = Mock()
        mock_reader.schema = schema
        mock_batch1, mock_batch2 = Mock(), Mock()
        mock_batch1.data = chunk1
        mock_batch2.data = chunk2
        mock_reader.read_chunk.side_effect = [
            mock_batch1,
            mock_batch2,
            StopIteration(),
        ]

        with patch(
            "arize._exporter.client.ArizeExportClient._get_stream_reader",
            return_value=(mock_reader, 4),
        ):
            export_client.export_to_parquet(
                path=output_path, **valid_export_params
            )

        assert tmp_path.joinpath("output.parquet").exists()

    def test_validates_path_type(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test that path type is validated."""
        with pytest.raises(TypeError, match=r"path.*must be a str"):
            export_client.export_to_parquet(path=123, **valid_export_params)  # type: ignore[arg-type]

    def test_empty_result_exits_early(
        self,
        export_client: ArizeExportClient,
        valid_export_params: dict,
        tmp_path: Path,
    ) -> None:
        """Test that empty result exits without creating file."""
        output_path = str(tmp_path / "output.parquet")

        with patch(
            "arize._exporter.client.ArizeExportClient._get_stream_reader",
            return_value=(None, 0),
        ):
            export_client.export_to_parquet(
                path=output_path, **valid_export_params
            )

        # File should not be created
        assert not tmp_path.joinpath("output.parquet").exists()

    def test_progress_bar_updates_correctly(
        self,
        export_client: ArizeExportClient,
        valid_export_params: dict,
        tmp_path: Path,
    ) -> None:
        """Test that progress bar updates with correct row counts."""
        output_path = str(tmp_path / "output.parquet")

        schema = pa.schema([("time", pa.int64())])
        chunk1 = pa.record_batch([[1000, 2000]], schema=schema)
        chunk2 = pa.record_batch([[3000]], schema=schema)

        mock_reader = Mock()
        mock_reader.schema = schema
        mock_batch1, mock_batch2 = Mock(), Mock()
        mock_batch1.data = chunk1
        mock_batch2.data = chunk2
        mock_reader.read_chunk.side_effect = [
            mock_batch1,
            mock_batch2,
            StopIteration(),
        ]

        with (
            patch(
                "arize._exporter.client.ArizeExportClient._get_stream_reader",
                return_value=(mock_reader, 3),
            ),
            patch("arize._exporter.client.tqdm") as mock_tqdm,
        ):
            mock_progress_bar = Mock()
            mock_tqdm.return_value = mock_progress_bar

            export_client.export_to_parquet(
                path=output_path, **valid_export_params
            )

            # Should be called twice with row counts
            assert mock_progress_bar.update.call_count == 2
            mock_progress_bar.update.assert_has_calls([call(2), call(1)])


@pytest.mark.unit
class TestHelperFunctions:
    """Test helper functions."""

    def test_get_progress_bar_with_records(self) -> None:
        """Test progress bar creation with records."""
        progress_bar = ArizeExportClient._get_progress_bar(100)
        assert progress_bar.total == 100
        progress_bar.close()

    def test_get_progress_bar_format(self) -> None:
        """Test progress bar has correct format."""
        progress_bar = ArizeExportClient._get_progress_bar(50)
        assert "exporting 50 rows" in progress_bar.desc
        progress_bar.close()

    @pytest.mark.parametrize(
        "num_references,include_timestamp,expected_predictions",
        [
            (1, True, ["pred1"]),
            (2, False, ["pred1", "pred2"]),
            (1, False, ["pred1"]),
        ],
    )
    def test_get_pb_similarity_search_params(
        self,
        num_references: int,
        include_timestamp: bool,
        expected_predictions: list[str],
    ) -> None:
        """Test conversion of SimilaritySearchParams to protobuf with various configs."""
        references = [
            SimilarityReference(
                prediction_id=f"pred{i + 1}",
                reference_column_name=f"ref_col{i + 1}",
                prediction_timestamp=datetime(2024, 1, 1)
                if include_timestamp
                else None,
            )
            for i in range(num_references)
        ]
        params = SimilaritySearchParams(
            search_column_name="embedding_col",
            threshold=0.85,
            references=references,
        )

        pb_params = _get_pb_similarity_search_params(params)

        assert pb_params.search_column_name == "embedding_col"
        assert pb_params.threshold == 0.85
        assert len(pb_params.references) == num_references
        assert [
            r.prediction_id for r in pb_params.references
        ] == expected_predictions


@pytest.mark.unit
class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    def test_full_export_workflow(
        self, export_client: ArizeExportClient, valid_export_params: dict
    ) -> None:
        """Test complete export workflow from request to DataFrame."""
        # Mock complete workflow
        mock_flight_info = Mock()
        mock_flight_info.total_records = 100
        mock_endpoint = Mock()
        mock_endpoint.ticket = Mock()
        mock_flight_info.endpoints = [mock_endpoint]
        export_client.flight_client.get_flight_info.return_value = (
            mock_flight_info
        )

        mock_reader = Mock()
        mock_reader.schema = pa.schema(
            [("time", pa.int64()), ("value", pa.float64())]
        )
        chunk = pa.table({"time": [1000, 2000, 3000], "value": [1.0, 2.0, 3.0]})
        mock_batch = Mock()
        mock_batch.data = chunk
        mock_reader.read_chunk.side_effect = [mock_batch, StopIteration()]
        export_client.flight_client.do_get.return_value = mock_reader

        result = export_client.export_to_df(**valid_export_params)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["time", "value"]

    def test_export_with_all_optional_params(
        self, export_client: ArizeExportClient
    ) -> None:
        """Test export with all optional parameters."""
        # Mock complete workflow
        mock_flight_info = Mock()
        mock_flight_info.total_records = 10
        mock_endpoint = Mock()
        mock_endpoint.ticket = Mock()
        mock_flight_info.endpoints = [mock_endpoint]
        export_client.flight_client.get_flight_info.return_value = (
            mock_flight_info
        )

        mock_reader = Mock()
        chunk = pa.table({"time": [1000], "value": [1.0]})
        mock_batch = Mock()
        mock_batch.data = chunk
        mock_reader.read_chunk.side_effect = [mock_batch, StopIteration()]
        export_client.flight_client.do_get.return_value = mock_reader

        result = export_client.export_to_df(
            space_id="space123",
            model_id="model456",
            environment=Environments.PRODUCTION,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            where="age > 30",
            columns=["col1", "col2"],
            model_version="v1.0",
            batch_id="batch123",
            include_actuals=True,
            stream_chunk_size=100,
            similarity_search_params=SimilaritySearchParams(
                search_column_name="embedding",
                threshold=0.8,
                references=[
                    SimilarityReference(
                        prediction_id="pred1",
                        reference_column_name="ref_col",
                    ),
                ],
            ),
        )

        assert isinstance(result, pd.DataFrame)
