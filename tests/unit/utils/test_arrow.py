"""Tests for Apache Arrow utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from arize.constants.pyarrow import DEFAULT_FLIGHT_BATCH_BUDGET_BYTES
from arize.exceptions.auth import AuthenticationError
from arize.exceptions.http import APIError
from arize.utils.arrow import (
    _append_to_pyarrow_metadata,
    _filesize,
    _maybe_log_project_url,
    _mktemp_in,
    _write_arrow_file,
    post_arrow_table,
    split_batches_by_byte_budget,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def sample_arrow_table() -> pa.Table:
    """Create a sample PyArrow table for testing."""
    return pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture
def mock_proto_schema() -> MagicMock:
    """Create a mock protobuf schema."""
    mock = MagicMock()
    mock.SerializeToString.return_value = b"mock_proto_bytes"
    return mock


@pytest.fixture
def mock_response() -> MagicMock:
    """Create a mock HTTP response."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"projectUrl": "https://app.arize.com/project/123"}
    return mock


@pytest.mark.unit
class TestPostArrowTable:
    """Test post_arrow_table function."""

    # Temp Directory Scenarios

    def test_tmp_dir_empty_creates_and_cleans_directory(
        self,
        sample_arrow_table: pa.Table,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should create temporary directory and clean it up when tmp_dir is empty."""
        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            result = post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=sample_arrow_table,
                proto_schema=mock_proto_schema,
                headers={"Authorization": "Bearer token"},
                timeout=30.0,
                verify=True,
                max_chunksize=1000,
                tmp_dir="",  # Empty means we own the directory
            )

            assert result == mock_response
            mock_post.assert_called_once()

    def test_tmp_dir_existing_creates_file_cleans_file_only(
        self,
        tmp_path: Path,
        sample_arrow_table: pa.Table,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should use provided directory and clean only the file."""
        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            result = post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=sample_arrow_table,
                proto_schema=mock_proto_schema,
                headers={"Authorization": "Bearer token"},
                timeout=30.0,
                verify=True,
                max_chunksize=1000,
                tmp_dir=str(tmp_path),
            )

            assert result == mock_response
            # Directory should still exist
            assert tmp_path.exists()

    def test_tmp_dir_file_path_writes_directly_no_cleanup(
        self,
        tmp_path: Path,
        sample_arrow_table: pa.Table,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should write directly to specified file path without cleanup."""
        file_path = tmp_path / "output.arrow"

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            result = post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=sample_arrow_table,
                proto_schema=mock_proto_schema,
                headers={"Authorization": "Bearer token"},
                timeout=30.0,
                verify=True,
                max_chunksize=1000,
                tmp_dir=str(file_path),
            )

            assert result == mock_response
            # File should still exist after upload
            assert file_path.exists()

    # Network/Upload

    def test_successful_post_returns_response(
        self,
        tmp_path: Path,
        sample_arrow_table: pa.Table,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should successfully post and return response."""
        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            result = post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=sample_arrow_table,
                proto_schema=mock_proto_schema,
                headers={"Authorization": "Bearer token"},
                timeout=30.0,
                verify=True,
                max_chunksize=1000,
                tmp_dir=str(tmp_path),
            )

            assert result == mock_response
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args.args[0] == "https://api.arize.com/upload"
            assert call_args.kwargs["timeout"] == 30.0
            assert call_args.kwargs["headers"] == {
                "Authorization": "Bearer token"
            }
            assert call_args.kwargs["verify"] is True

    def test_post_with_custom_headers(
        self,
        tmp_path: Path,
        sample_arrow_table: pa.Table,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should pass custom headers to request."""
        custom_headers = {
            "Authorization": "Bearer token",
            "X-Custom-Header": "value",
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=sample_arrow_table,
                proto_schema=mock_proto_schema,
                headers=custom_headers,
                timeout=30.0,
                verify=True,
                max_chunksize=1000,
                tmp_dir=str(tmp_path),
            )

            call_args = mock_post.call_args
            assert call_args.kwargs["headers"] == custom_headers

    def test_post_with_timeout(
        self,
        tmp_path: Path,
        sample_arrow_table: pa.Table,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should respect timeout parameter."""
        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=sample_arrow_table,
                proto_schema=mock_proto_schema,
                headers={"Authorization": "Bearer token"},
                timeout=60.0,
                verify=True,
                max_chunksize=1000,
                tmp_dir=str(tmp_path),
            )

            call_args = mock_post.call_args
            assert call_args.kwargs["timeout"] == 60.0

    def test_post_with_verify_false(
        self,
        tmp_path: Path,
        sample_arrow_table: pa.Table,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should disable SSL verification when verify=False."""
        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=sample_arrow_table,
                proto_schema=mock_proto_schema,
                headers={"Authorization": "Bearer token"},
                timeout=30.0,
                verify=False,
                max_chunksize=1000,
                tmp_dir=str(tmp_path),
            )

            call_args = mock_post.call_args
            assert call_args.kwargs["verify"] is False

    # Schema Handling

    def test_appends_proto_schema_to_metadata(
        self,
        tmp_path: Path,
        sample_arrow_table: pa.Table,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should append base64-encoded proto schema to arrow metadata."""
        with (
            patch("requests.post") as mock_post,
            patch("arize.utils.arrow._write_arrow_file") as mock_write,
        ):
            mock_post.return_value = mock_response

            post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=sample_arrow_table,
                proto_schema=mock_proto_schema,
                headers={"Authorization": "Bearer token"},
                timeout=30.0,
                verify=True,
                max_chunksize=1000,
                tmp_dir=str(tmp_path),
            )

            # Verify schema was serialized
            mock_proto_schema.SerializeToString.assert_called_once()

            # Verify write was called with modified schema
            mock_write.assert_called_once()
            call_args = mock_write.call_args
            pa_schema = call_args.args[2]
            assert b"arize-schema" in pa_schema.metadata

    def test_schema_metadata_not_overwritten(
        self,
        tmp_path: Path,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should not overwrite existing metadata in schema."""
        # Create table with existing metadata
        schema = pa.schema([("a", pa.int64())]).with_metadata(
            {"existing-key": b"existing-value"}
        )
        table_with_metadata = pa.table({"a": [1, 2, 3]}, schema=schema)

        with (
            patch("requests.post") as mock_post,
            patch("arize.utils.arrow._write_arrow_file") as mock_write,
        ):
            mock_post.return_value = mock_response

            post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=table_with_metadata,
                proto_schema=mock_proto_schema,
                headers={"Authorization": "Bearer token"},
                timeout=30.0,
                verify=True,
                max_chunksize=1000,
                tmp_dir=str(tmp_path),
            )

            # Verify existing metadata is preserved
            mock_write.assert_called_once()
            call_args = mock_write.call_args
            pa_schema = call_args.args[2]
            assert b"existing-key" in pa_schema.metadata
            assert b"arize-schema" in pa_schema.metadata

    # Error Handling

    def test_cleanup_on_post_failure(
        self,
        tmp_path: Path,
        sample_arrow_table: pa.Table,
        mock_proto_schema: MagicMock,
    ) -> None:
        """Should clean up temporary file when post fails."""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Upload failed")

            with pytest.raises(Exception, match="Upload failed"):
                post_arrow_table(
                    files_url="https://api.arize.com/upload",
                    pa_table=sample_arrow_table,
                    proto_schema=mock_proto_schema,
                    headers={"Authorization": "Bearer token"},
                    timeout=30.0,
                    verify=True,
                    max_chunksize=1000,
                    tmp_dir=str(tmp_path),
                )

            # Verify cleanup was attempted (file should not exist)
            arrow_files = list(tmp_path.glob("arize-*.arrow"))
            assert len(arrow_files) == 0

    def test_cleanup_on_write_failure(
        self, sample_arrow_table: pa.Table, mock_proto_schema: MagicMock
    ) -> None:
        """Should clean up temporary directory when write fails."""
        with patch("arize.utils.arrow._write_arrow_file") as mock_write:
            mock_write.side_effect = Exception("Write failed")

            with pytest.raises(Exception, match="Write failed"):
                post_arrow_table(
                    files_url="https://api.arize.com/upload",
                    pa_table=sample_arrow_table,
                    proto_schema=mock_proto_schema,
                    headers={"Authorization": "Bearer token"},
                    timeout=30.0,
                    verify=True,
                    max_chunksize=1000,
                    tmp_dir="",  # Empty means we own the directory
                )

    @pytest.mark.parametrize("status_code", [401, 403])
    def test_raises_authentication_error_on_auth_failure(
        self,
        tmp_path: Path,
        sample_arrow_table: pa.Table,
        mock_proto_schema: MagicMock,
        status_code: int,
    ) -> None:
        """Should raise AuthenticationError immediately on 401/403."""
        auth_response = MagicMock()
        auth_response.status_code = status_code
        auth_response.text = '{"error":"space key or api key is invalid"}'

        with patch("requests.post") as mock_post:
            mock_post.return_value = auth_response

            with pytest.raises(AuthenticationError) as exc_info:
                post_arrow_table(
                    files_url="https://api.arize.com/upload",
                    pa_table=sample_arrow_table,
                    proto_schema=mock_proto_schema,
                    headers={"Authorization": "Bearer bad-key"},
                    timeout=30.0,
                    verify=True,
                    max_chunksize=1000,
                    tmp_dir=str(tmp_path),
                )

            assert exc_info.value.status_code == status_code
            assert "Verify your API key" in str(exc_info.value)

    @pytest.mark.parametrize("status_code", [400, 422, 429, 500, 503])
    def test_raises_api_error_on_non_2xx(
        self,
        tmp_path: Path,
        sample_arrow_table: pa.Table,
        mock_proto_schema: MagicMock,
        status_code: int,
    ) -> None:
        """Should raise APIError immediately on any non-2xx response (other than 401/403)."""
        error_response = MagicMock()
        error_response.status_code = status_code
        error_response.text = "server error"

        with patch("requests.post") as mock_post:
            mock_post.return_value = error_response

            with pytest.raises(APIError) as exc_info:
                post_arrow_table(
                    files_url="https://api.arize.com/upload",
                    pa_table=sample_arrow_table,
                    proto_schema=mock_proto_schema,
                    headers={"Authorization": "Bearer token"},
                    timeout=30.0,
                    verify=True,
                    max_chunksize=1000,
                    tmp_dir=str(tmp_path),
                )

            assert exc_info.value.status_code == status_code

    def test_logs_project_url_on_success(
        self,
        tmp_path: Path,
        sample_arrow_table: pa.Table,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should log project URL on successful upload."""
        with (
            patch("requests.post") as mock_post,
            patch("arize.utils.arrow._maybe_log_project_url") as mock_log_url,
        ):
            mock_post.return_value = mock_response

            post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=sample_arrow_table,
                proto_schema=mock_proto_schema,
                headers={"Authorization": "Bearer token"},
                timeout=30.0,
                verify=True,
                max_chunksize=1000,
                tmp_dir=str(tmp_path),
            )

            mock_log_url.assert_called_once_with(mock_response)

    # Edge Cases

    def test_empty_table(
        self,
        tmp_path: Path,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should handle empty table with 0 rows."""
        empty_table = pa.table({"a": [], "b": []})

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            result = post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=empty_table,
                proto_schema=mock_proto_schema,
                headers={"Authorization": "Bearer token"},
                timeout=30.0,
                verify=True,
                max_chunksize=1000,
                tmp_dir=str(tmp_path),
            )

            assert result == mock_response

    def test_large_table_with_chunking(
        self,
        tmp_path: Path,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should handle large table with chunking."""
        # Create a large table
        large_table = pa.table({"a": list(range(10000)), "b": ["x"] * 10000})

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            result = post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=large_table,
                proto_schema=mock_proto_schema,
                headers={"Authorization": "Bearer token"},
                timeout=30.0,
                verify=True,
                max_chunksize=100,  # Small chunk size to test chunking
                tmp_dir=str(tmp_path),
            )

            assert result == mock_response

    def test_various_column_types(
        self,
        tmp_path: Path,
        mock_proto_schema: MagicMock,
        mock_response: MagicMock,
    ) -> None:
        """Should handle various column types."""
        import datetime

        table = pa.table(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "string_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "timestamp_col": [
                    datetime.datetime(2020, 1, 1),
                    datetime.datetime(2020, 1, 2),
                    datetime.datetime(2020, 1, 3),
                ],
            }
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_response

            result = post_arrow_table(
                files_url="https://api.arize.com/upload",
                pa_table=table,
                proto_schema=mock_proto_schema,
                headers={"Authorization": "Bearer token"},
                timeout=30.0,
                verify=True,
                max_chunksize=1000,
                tmp_dir=str(tmp_path),
            )

            assert result == mock_response


@pytest.mark.unit
class TestAppendToPyarrowMetadata:
    """Test _append_to_pyarrow_metadata function."""

    def test_appends_to_empty_metadata(self) -> None:
        """Should initialize empty dict and append metadata."""
        schema = pa.schema([("a", pa.int64())])
        new_metadata = {"key1": b"value1", "key2": b"value2"}

        result = _append_to_pyarrow_metadata(schema, new_metadata)

        assert b"key1" in result.metadata
        assert b"key2" in result.metadata
        assert result.metadata[b"key1"] == b"value1"

    def test_appends_to_existing_metadata(self) -> None:
        """Should merge with existing metadata."""
        schema = pa.schema([("a", pa.int64())]).with_metadata(
            {"existing": b"value"}
        )
        new_metadata = {"new_key": b"new_value"}

        result = _append_to_pyarrow_metadata(schema, new_metadata)

        assert b"existing" in result.metadata
        assert b"new_key" in result.metadata

    def test_raises_on_conflicting_keys(self) -> None:
        """Should raise KeyError when keys conflict."""
        schema = pa.schema([("a", pa.int64())]).with_metadata(
            {b"conflict": b"value"}
        )
        new_metadata = {b"conflict": b"new_value"}

        with pytest.raises(KeyError, match="conflicting keys"):
            _append_to_pyarrow_metadata(schema, new_metadata)

    def test_handles_bytes_in_metadata(self) -> None:
        """Should handle bytes values in metadata."""
        schema = pa.schema([("a", pa.int64())])
        new_metadata = {"bytes_key": b"bytes_value"}

        result = _append_to_pyarrow_metadata(schema, new_metadata)

        assert result.metadata[b"bytes_key"] == b"bytes_value"


@pytest.mark.unit
class TestWriteArrowFile:
    """Test _write_arrow_file function."""

    def test_writes_valid_arrow_file(
        self, tmp_path: Path, sample_arrow_table: pa.Table
    ) -> None:
        """Should write a valid arrow file that can be read back."""
        file_path = tmp_path / "test.arrow"
        schema = sample_arrow_table.schema

        _write_arrow_file(str(file_path), sample_arrow_table, schema, 1000)

        assert file_path.exists()

        # Verify file can be read back
        with (
            pa.OSFile(str(file_path), mode="rb") as source,
            pa.ipc.RecordBatchStreamReader(source) as reader,
        ):
            read_table = reader.read_all()
            assert read_table.num_rows == sample_arrow_table.num_rows
            assert read_table.num_columns == sample_arrow_table.num_columns

    def test_chunks_large_table(self, tmp_path: Path) -> None:
        """Should respect max_chunksize parameter."""
        large_table = pa.table({"a": list(range(1000))})
        file_path = tmp_path / "chunked.arrow"
        schema = large_table.schema

        _write_arrow_file(str(file_path), large_table, schema, 100)

        assert file_path.exists()

        # Verify file was written with chunks
        with (
            pa.OSFile(str(file_path), mode="rb") as source,
            pa.ipc.RecordBatchStreamReader(source) as reader,
        ):
            read_table = reader.read_all()
            assert read_table.num_rows == 1000

    def test_raises_on_write_permission_error(
        self, sample_arrow_table: pa.Table
    ) -> None:
        """Should raise exception when write permission is denied."""
        schema = sample_arrow_table.schema

        with pytest.raises(Exception):
            # Try to write to a non-existent directory
            _write_arrow_file(
                "/nonexistent/path/file.arrow", sample_arrow_table, schema, 1000
            )


@pytest.mark.unit
class TestMaybeLogProjectUrl:
    """Test _maybe_log_project_url function."""

    def test_logs_project_url_on_success(
        self, mock_response: MagicMock
    ) -> None:
        """Should log project URL when extraction succeeds."""
        with (
            patch("arize.utils.arrow.get_arize_project_url") as mock_get_url,
            patch("arize.utils.arrow.logger.info") as mock_info,
        ):
            mock_get_url.return_value = "https://app.arize.com/project/123"

            _maybe_log_project_url(mock_response)

            mock_get_url.assert_called_once_with(mock_response)
            mock_info.assert_called_once()
            assert "Success" in str(mock_info.call_args)

    def test_logs_nothing_on_extraction_failure(
        self, mock_response: MagicMock
    ) -> None:
        """Should log a diagnostic when the response has no project URL."""
        with (
            patch("arize.utils.arrow.get_arize_project_url") as mock_get_url,
            patch("arize.utils.arrow.logger.debug") as mock_debug,
        ):
            mock_get_url.return_value = None

            _maybe_log_project_url(mock_response)

            mock_get_url.assert_called_once_with(mock_response)
            mock_debug.assert_called_once()
            assert "without a project URL" in str(mock_debug.call_args)

    def test_never_raises_exception(self, mock_response: MagicMock) -> None:
        """Should never raise exception even if extraction fails."""
        with (
            patch("arize.utils.arrow.get_arize_project_url") as mock_get_url,
            patch("arize.utils.arrow.logger.warning") as mock_warning,
        ):
            mock_get_url.side_effect = Exception("Extraction failed")

            # Should not raise
            _maybe_log_project_url(mock_response)

            mock_warning.assert_called_once()
            assert "Failed to get project URL" in str(mock_warning.call_args)


@pytest.mark.unit
class TestMktempIn:
    """Test _mktemp_in function."""

    def test_creates_unique_temp_file(self, tmp_path: Path) -> None:
        """Should create unique temp files on multiple calls."""
        file1 = _mktemp_in(str(tmp_path))
        file2 = _mktemp_in(str(tmp_path))

        assert file1 != file2
        assert Path(file1).exists()
        assert Path(file2).exists()

    def test_file_exists_after_creation(self, tmp_path: Path) -> None:
        """Should create file that exists and is closed."""
        file_path = _mktemp_in(str(tmp_path))

        assert Path(file_path).exists()
        # Should be able to open and write to it
        with open(file_path, "w") as f:
            f.write("test")

    def test_raises_on_invalid_directory(self) -> None:
        """Should raise exception when directory doesn't exist."""
        with pytest.raises(Exception):
            _mktemp_in("/nonexistent/directory")


@pytest.mark.unit
class TestFilesize:
    """Test _filesize function."""

    def test_returns_file_size_in_bytes(self, tmp_path: Path) -> None:
        """Should return correct file size in bytes."""
        file_path = tmp_path / "test.txt"
        content = "test content"
        file_path.write_text(content)

        size = _filesize(str(file_path))

        assert size == len(content.encode())

    def test_returns_negative_one_on_error(self) -> None:
        """Should return -1 when file doesn't exist or can't be accessed."""
        size = _filesize("/nonexistent/file.txt")

        assert size == -1


def _skewed_table(widths: list[int]) -> pa.Table:
    """Build a one-column table whose row `i` holds `widths[i]` bytes of text."""
    return pa.table({"payload": ["x" * width for width in widths]})


def _split_table(
    table: pa.Table,
    max_batch_bytes: int = DEFAULT_FLIGHT_BATCH_BUDGET_BYTES,
    max_batch_rows: int | None = None,
) -> list[pa.RecordBatch]:
    return list(
        split_batches_by_byte_budget(
            table.to_batches(), max_batch_bytes, max_batch_rows
        )
    )


@pytest.mark.unit
class TestSplitBatchesByByteBudget:
    """Test split_batches_by_byte_budget function."""

    def test_empty_table_yields_nothing(self) -> None:
        """Should yield no batches for a table with no rows."""
        table = _skewed_table([10]).schema.empty_table()

        assert _split_table(table) == []

    def test_single_row_table_yields_one_batch(self) -> None:
        """Should yield a single batch for a single-row table."""
        table = _skewed_table([100])

        batches = _split_table(table)

        assert len(batches) == 1
        assert batches[0].num_rows == 1

    def test_tiny_rows_fit_in_one_batch(self) -> None:
        """Should keep a table well under the budget as one batch."""
        table = _skewed_table([4] * 5_000)

        batches = _split_table(table)

        assert len(batches) == 1
        assert batches[0].num_rows == 5_000

    def test_splits_when_running_size_crosses_budget(self) -> None:
        """Should cut a batch once accumulated rows cross the byte budget."""
        table = _skewed_table([1_000] * 100)

        batches = _split_table(table, max_batch_bytes=10_000)

        assert len(batches) > 1
        assert all(batch.nbytes <= 10_000 for batch in batches)

    def test_preserves_all_rows_in_order(self) -> None:
        """Should reproduce the original table when batches are concatenated."""
        table = _skewed_table([50, 900, 50, 5_000, 50, 120, 3_000])

        batches = _split_table(table, max_batch_bytes=2_000)

        assert pa.Table.from_batches(batches, schema=table.schema).equals(table)

    def test_extreme_skew_keeps_batches_within_budget(self) -> None:
        """Should bound batches by bytes when one row is 100x the median."""
        widths = [1_000] * 50 + [100_000] + [1_000] * 50
        table = _skewed_table(widths)

        batches = _split_table(table, max_batch_bytes=20_000)

        oversized = [batch for batch in batches if batch.nbytes > 20_000]
        assert [batch.num_rows for batch in oversized] == [1]
        assert sum(batch.num_rows for batch in batches) == len(widths)

    def test_row_larger_than_budget_is_emitted_alone(self) -> None:
        """Should emit a row that exceeds the budget as its own batch."""
        table = _skewed_table([100, 50_000, 100])

        batches = _split_table(table, max_batch_bytes=1_000)

        assert [batch.num_rows for batch in batches] == [1, 1, 1]
        assert batches[1].nbytes > 1_000

    def test_row_over_server_limit_is_yielded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should yield a row over the server limit so the server rejects it."""
        monkeypatch.setattr(
            "arize.utils.arrow.FLIGHT_SERVER_MAX_MESSAGE_BYTES", 10_000
        )
        table = _skewed_table([100, 50_000])

        batches = _split_table(table, max_batch_bytes=1_000)

        assert [batch.num_rows for batch in batches] == [1, 1]
        assert batches[1].nbytes > 10_000

    def test_budget_above_server_limit_raises_before_reading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should reject a budget above the server limit before any batch is read."""
        monkeypatch.setattr(
            "arize.utils.arrow.FLIGHT_SERVER_MAX_MESSAGE_BYTES", 10_000
        )

        def source() -> Iterator[pa.RecordBatch]:
            raise AssertionError("source was read")
            yield

        with pytest.raises(ValueError, match="max_batch_bytes"):
            split_batches_by_byte_budget(source(), max_batch_bytes=1_000_000)

    def test_row_ceiling_caps_batch_rows(self) -> None:
        """Should never exceed max_batch_rows even when bytes allow more."""
        table = _skewed_table([4] * 1_000)

        batches = _split_table(table, max_batch_rows=250)

        assert [batch.num_rows for batch in batches] == [250] * 4

    def test_byte_budget_applies_under_row_ceiling(self) -> None:
        """Should still cut by bytes when the row ceiling is generous."""
        table = _skewed_table([1_000] * 100)

        batches = _split_table(
            table, max_batch_bytes=10_000, max_batch_rows=100_000
        )

        assert len(batches) > 1
        assert all(batch.nbytes <= 10_000 for batch in batches)

    def test_dictionary_column_counted_once_per_batch(self) -> None:
        """Should size a dictionary column by the slice, not by each row."""
        labels = pa.array(["a" * 4_000, "b" * 4_000] * 500).dictionary_encode()
        table = pa.table({"label": labels, "payload": ["x" * 100] * 1_000})

        batches = _split_table(table, max_batch_bytes=20_000)

        assert all(batch.nbytes <= 20_000 for batch in batches)
        assert sum(batch.num_rows for batch in batches) == 1_000
        assert max(batch.num_rows for batch in batches) > 1

    def test_skips_empty_chunks(self) -> None:
        """Should drop the zero-row batches an empty column chunk produces."""
        table = pa.table(
            {
                "payload": pa.chunked_array(
                    [pa.array([], type=pa.string()), pa.array(["a", "b"])]
                )
            }
        )

        batches = _split_table(table)

        assert [batch.num_rows for batch in batches] == [2]

    def test_multi_chunk_table_is_split(self) -> None:
        """Should handle a table whose columns already hold several chunks."""
        table = pa.concat_tables(
            [_skewed_table([1_000] * 20), _skewed_table([1_000] * 20)]
        )

        batches = _split_table(table, max_batch_bytes=5_000)

        assert sum(batch.num_rows for batch in batches) == 40
        assert all(batch.nbytes <= 5_000 for batch in batches)

    @pytest.mark.parametrize(
        ("max_batch_bytes", "max_batch_rows"),
        [(0, None), (-1, None), (1_000, 0), (1_000, -5)],
    )
    def test_rejects_non_positive_bounds(
        self, max_batch_bytes: int, max_batch_rows: int | None
    ) -> None:
        """Should reject a budget or row ceiling below one."""
        table = _skewed_table([10])

        with pytest.raises(ValueError):
            _split_table(
                table,
                max_batch_bytes=max_batch_bytes,
                max_batch_rows=max_batch_rows,
            )

    def test_is_lazy(self) -> None:
        """Should not pull a batch from the source until the caller asks."""
        pulled = []

        def source() -> Iterator[pa.RecordBatch]:
            for width in (100, 100):
                pulled.append(width)
                yield _skewed_table([width]).to_batches()[0]

        out = split_batches_by_byte_budget(source())

        assert pulled == []
        next(out)
        assert pulled == [100]

    def test_splits_each_batch_by_bytes_and_rows(self) -> None:
        """Should bound every output batch by the byte budget and row ceiling."""
        table = _skewed_table([1_000] * 40 + [4] * 40)
        source = table.to_batches(max_chunksize=50)

        batches = list(
            split_batches_by_byte_budget(
                source, max_batch_bytes=10_000, max_batch_rows=20
            )
        )

        assert all(b.nbytes <= 10_000 and b.num_rows <= 20 for b in batches)
        assert pa.Table.from_batches(batches, schema=table.schema).equals(table)
