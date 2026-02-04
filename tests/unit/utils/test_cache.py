"""Tests for caching utilities."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from arize.utils.cache import (
    _get_abs_file_path,
    _get_cache_key,
    cache_resource,
    load_cached_resource,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture
def sample_timestamp() -> datetime:
    """Create a sample timestamp for testing."""
    return datetime(2024, 1, 15, 10, 30, 45)


@pytest.mark.unit
class TestLoadCachedResource:
    """Test load_cached_resource function."""

    def test_returns_dataframe_when_file_exists(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Should return DataFrame when cache file exists."""
        # First cache the resource
        cache_resource(str(tmp_path), "dataset", "id123", None, sample_df)

        # Then load it
        result = load_cached_resource(str(tmp_path), "dataset", "id123", None)
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_df)

    def test_returns_none_when_file_not_found(self, tmp_path: Path) -> None:
        """Should return None when cache file doesn't exist."""
        result = load_cached_resource(
            str(tmp_path), "dataset", "nonexistent", None
        )
        assert result is None

    def test_returns_none_when_parquet_corrupted(self, tmp_path: Path) -> None:
        """Should return None and log warning when parquet file is corrupted."""
        # Create a corrupted file
        cache_dir = tmp_path / "dataset"
        cache_dir.mkdir(parents=True)
        corrupted_file = cache_dir / "dataset_id123.parquet"
        corrupted_file.write_text("not a valid parquet file")

        with patch("arize.utils.cache.logger.warning") as mock_warning:
            result = load_cached_resource(
                str(tmp_path), "dataset", "id123", None
            )
            assert result is None
            mock_warning.assert_called_once()
            assert "Failed to load cached resource" in str(
                mock_warning.call_args
            )

    def test_expands_home_directory(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Should expand ~ in cache_dir path."""
        # Use a path relative to tmp_path but reference it with ~
        # We'll mock expanduser to return tmp_path
        with patch("pathlib.Path.expanduser") as mock_expand:
            mock_expand.return_value = Path(str(tmp_path))
            cache_resource("~/cache", "dataset", "id123", None, sample_df)

            # Verify expanduser was called
            mock_expand.assert_called()

    def test_uses_resource_updated_at_in_filename(
        self,
        tmp_path: Path,
        sample_df: pd.DataFrame,
        sample_timestamp: datetime,
    ) -> None:
        """Should include timestamp in filename when resource_updated_at is provided."""
        cache_resource(
            str(tmp_path), "dataset", "id123", sample_timestamp, sample_df
        )

        # Load with same timestamp
        result = load_cached_resource(
            str(tmp_path), "dataset", "id123", sample_timestamp
        )
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_df)

        # Verify the file has the timestamp in the name
        cache_file = (
            tmp_path / "dataset" / "dataset_id123_20240115T103045.parquet"
        )
        assert cache_file.exists()

    def test_ignores_resource_updated_at_when_none(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Should not include timestamp in filename when resource_updated_at is None."""
        cache_resource(str(tmp_path), "dataset", "id123", None, sample_df)

        result = load_cached_resource(str(tmp_path), "dataset", "id123", None)
        assert result is not None

        # Verify the file does NOT have a timestamp in the name
        cache_file = tmp_path / "dataset" / "dataset_id123.parquet"
        assert cache_file.exists()

    def test_logs_warning_on_read_failure(self, tmp_path: Path) -> None:
        """Should log warning when read fails."""
        with (
            patch("arize.utils.cache.pd.read_parquet") as mock_read,
            patch("arize.utils.cache.logger.warning") as mock_warning,
        ):
            mock_read.side_effect = Exception("Read error")

            # Create the file so it exists but reading fails
            cache_dir = tmp_path / "dataset"
            cache_dir.mkdir(parents=True)
            (cache_dir / "dataset_id123.parquet").touch()

            result = load_cached_resource(
                str(tmp_path), "dataset", "id123", None
            )
            assert result is None
            mock_warning.assert_called_once()

    def test_handles_subdirectory_structure(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Should use resource name as subdirectory."""
        cache_resource(str(tmp_path), "experiments", "exp456", None, sample_df)

        # Verify directory structure
        cache_file = tmp_path / "experiments" / "experiments_exp456.parquet"
        assert cache_file.exists()

        result = load_cached_resource(
            str(tmp_path), "experiments", "exp456", None
        )
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_df)


@pytest.mark.unit
class TestCacheResource:
    """Test cache_resource function."""

    def test_saves_dataframe_to_parquet(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Should save DataFrame to parquet file."""
        cache_resource(str(tmp_path), "dataset", "id123", None, sample_df)

        cache_file = tmp_path / "dataset" / "dataset_id123.parquet"
        assert cache_file.exists()

        loaded = pd.read_parquet(cache_file)
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_creates_parent_directories(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Should create parent directories if they don't exist."""
        nested_path = tmp_path / "level1" / "level2"
        cache_resource(str(nested_path), "dataset", "id123", None, sample_df)

        cache_file = nested_path / "dataset" / "dataset_id123.parquet"
        assert cache_file.exists()

    def test_uses_resource_as_subdirectory(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Should create subdirectory with resource name."""
        cache_resource(str(tmp_path), "my_resource", "id123", None, sample_df)

        cache_file = tmp_path / "my_resource" / "my_resource_id123.parquet"
        assert cache_file.exists()

    def test_logs_debug_message(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Should log debug message when caching."""
        with patch("arize.utils.cache.logger.debug") as mock_debug:
            cache_resource(str(tmp_path), "dataset", "id123", None, sample_df)
            mock_debug.assert_called_once()
            assert "Cached resource to" in str(mock_debug.call_args)

    def test_handles_existing_directory(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Should not error if directory already exists."""
        # Create directory first
        cache_dir = tmp_path / "dataset"
        cache_dir.mkdir(parents=True)

        # Should not raise error
        cache_resource(str(tmp_path), "dataset", "id123", None, sample_df)

        cache_file = cache_dir / "dataset_id123.parquet"
        assert cache_file.exists()

    def test_expands_home_directory(
        self, tmp_path: Path, sample_df: pd.DataFrame
    ) -> None:
        """Should expand ~ in cache_dir path."""
        with patch("pathlib.Path.expanduser") as mock_expand:
            mock_expand.return_value = Path(str(tmp_path))
            cache_resource("~/cache", "dataset", "id123", None, sample_df)

            # Verify expanduser was called
            mock_expand.assert_called()

    def test_uses_resource_updated_at_in_filename(
        self,
        tmp_path: Path,
        sample_df: pd.DataFrame,
        sample_timestamp: datetime,
    ) -> None:
        """Should include timestamp in filename when resource_updated_at is provided."""
        cache_resource(
            str(tmp_path), "dataset", "id123", sample_timestamp, sample_df
        )

        cache_file = (
            tmp_path / "dataset" / "dataset_id123_20240115T103045.parquet"
        )
        assert cache_file.exists()


@pytest.mark.unit
class TestGetCacheKey:
    """Test _get_cache_key function."""

    def test_format_without_timestamp(self) -> None:
        """Should format key as resource_resourceId when no timestamp."""
        key = _get_cache_key("dataset", "id123", None)
        assert key == "dataset_id123"

    def test_format_with_timestamp(self, sample_timestamp: datetime) -> None:
        """Should include formatted timestamp when provided."""
        key = _get_cache_key("dataset", "id123", sample_timestamp)
        assert key == "dataset_id123_20240115T103045"

    def test_timestamp_format(self) -> None:
        """Should format timestamp as YYYYMMDDTHHMMSS."""
        timestamp = datetime(2024, 12, 31, 23, 59, 59)
        key = _get_cache_key("dataset", "id123", timestamp)
        assert key == "dataset_id123_20241231T235959"


@pytest.mark.unit
class TestGetAbsFilePath:
    """Test _get_abs_file_path function."""

    def test_expands_home_directory(self) -> None:
        """Should expand ~ in directory path."""
        result = _get_abs_file_path("~/cache", "test.parquet", None)
        assert "~" not in str(result)
        assert result.is_absolute()

    def test_joins_subdirectory(self, tmp_path: Path) -> None:
        """Should join subdirectory to base path."""
        result = _get_abs_file_path(str(tmp_path), "test.parquet", "dataset")
        expected = tmp_path / "dataset" / "test.parquet"
        assert result == expected.resolve()

    def test_without_subdirectory(self, tmp_path: Path) -> None:
        """Should work without subdirectory."""
        result = _get_abs_file_path(str(tmp_path), "test.parquet", None)
        expected = tmp_path / "test.parquet"
        assert result == expected.resolve()

    def test_resolves_relative_paths(self) -> None:
        """Should resolve relative path components."""
        result = _get_abs_file_path("./cache/../data", "test.parquet", None)
        assert result.is_absolute()
        assert ".." not in str(result)
