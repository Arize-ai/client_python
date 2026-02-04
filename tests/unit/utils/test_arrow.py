"""Tests for Apache Arrow utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from arize.utils.arrow import (
    _append_to_pyarrow_metadata,
    _filesize,
    _maybe_log_project_url,
    _mktemp_in,
    _write_arrow_file,
    post_arrow_table,
)


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
        """Should not log when URL extraction returns None."""
        with (
            patch("arize.utils.arrow.get_arize_project_url") as mock_get_url,
            patch("arize.utils.arrow.logger.info") as mock_info,
        ):
            mock_get_url.return_value = None

            _maybe_log_project_url(mock_response)

            mock_get_url.assert_called_once_with(mock_response)
            mock_info.assert_not_called()

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
