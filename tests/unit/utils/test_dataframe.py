"""Tests for DataFrame manipulation and validation utilities."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from arize.utils.dataframe import (
    remove_extraneous_columns,
    reset_dataframe_index,
)


@pytest.mark.unit
class TestResetDataframeIndex:
    """Test reset_dataframe_index function."""

    def test_range_index_not_reset(self) -> None:
        """Should not reset DataFrame with RangeIndex."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        original_index = df.index
        reset_dataframe_index(df)
        assert isinstance(df.index, pd.RangeIndex)
        pd.testing.assert_index_equal(df.index, original_index)

    def test_non_range_index_gets_reset(self) -> None:
        """Should reset DataFrame with non-RangeIndex."""
        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "y", "z"]
        )
        reset_dataframe_index(df)
        assert isinstance(df.index, pd.RangeIndex)
        assert df.index.tolist() == [0, 1, 2]

    def test_named_index_dropped_if_in_columns(self) -> None:
        """Should drop named index if name already exists in columns."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df = df.set_index("a")
        reset_dataframe_index(df)
        assert isinstance(df.index, pd.RangeIndex)
        assert "a" in df.columns
        # Should not have duplicate 'a' column
        assert list(df.columns).count("a") == 1

    def test_named_index_not_in_columns_kept(self) -> None:
        """Should keep named index as column if not in DataFrame columns."""
        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "y", "z"]
        )
        df.index.name = "my_index"
        reset_dataframe_index(df)
        assert isinstance(df.index, pd.RangeIndex)
        assert "my_index" in df.columns

    def test_modifies_dataframe_in_place(self) -> None:
        """Should modify DataFrame in-place."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
        original_id = id(df)
        reset_dataframe_index(df)
        assert id(df) == original_id

    def test_multiindex_gets_reset(self) -> None:
        """Should reset DataFrame with MultiIndex."""
        arrays = [["a", "a", "b"], [1, 2, 1]]
        index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
        df = pd.DataFrame({"value": [10, 20, 30]}, index=index)
        reset_dataframe_index(df)
        assert isinstance(df.index, pd.RangeIndex)

    def test_datetime_index_gets_reset(self) -> None:
        """Should reset DataFrame with DatetimeIndex."""
        dates = pd.date_range("2020-01-01", periods=3)
        df = pd.DataFrame({"a": [1, 2, 3]}, index=dates)
        reset_dataframe_index(df)
        assert isinstance(df.index, pd.RangeIndex)

    def test_empty_dataframe_with_range_index(self) -> None:
        """Should handle empty DataFrame with RangeIndex."""
        df = pd.DataFrame()
        reset_dataframe_index(df)
        assert isinstance(df.index, pd.RangeIndex)


@pytest.mark.unit
class TestRemoveExtraneousColumns:
    """Test remove_extraneous_columns function."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "col_a": [1, 2, 3],
                "col_b": [4, 5, 6],
                "col_c": [7, 8, 9],
                "other_x": [10, 11, 12],
                "other_y": [13, 14, 15],
            }
        )

    @pytest.fixture
    def mock_schema(self) -> MagicMock:
        """Create mock schema for testing."""
        schema = MagicMock()
        schema.get_used_columns.return_value = ["col_a", "col_b"]
        return schema

    def test_filter_by_schema(
        self, sample_df: pd.DataFrame, mock_schema: MagicMock
    ) -> None:
        """Should filter columns based on schema."""
        result = remove_extraneous_columns(sample_df, schema=mock_schema)
        assert set(result.columns) == {"col_a", "col_b"}
        mock_schema.get_used_columns.assert_called_once()

    def test_filter_by_column_list(self, sample_df: pd.DataFrame) -> None:
        """Should filter columns based on explicit list."""
        result = remove_extraneous_columns(
            sample_df, column_list=["col_a", "col_c"]
        )
        assert set(result.columns) == {"col_a", "col_c"}

    def test_filter_by_regex(self, sample_df: pd.DataFrame) -> None:
        """Should filter columns based on regex pattern."""
        result = remove_extraneous_columns(sample_df, regex=r"^other_")
        assert set(result.columns) == {"other_x", "other_y"}

    def test_filter_by_schema_and_column_list(
        self, sample_df: pd.DataFrame, mock_schema: MagicMock
    ) -> None:
        """Should combine schema and column list filters."""
        result = remove_extraneous_columns(
            sample_df, schema=mock_schema, column_list=["col_c"]
        )
        assert set(result.columns) == {"col_a", "col_b", "col_c"}

    def test_filter_by_all_parameters(
        self, sample_df: pd.DataFrame, mock_schema: MagicMock
    ) -> None:
        """Should combine schema, column list, and regex filters."""
        result = remove_extraneous_columns(
            sample_df,
            schema=mock_schema,
            column_list=["col_c"],
            regex=r"^other_x$",
        )
        assert set(result.columns) == {"col_a", "col_b", "col_c", "other_x"}

    def test_no_filters_returns_empty(self, sample_df: pd.DataFrame) -> None:
        """Should return empty DataFrame when no filters specified."""
        result = remove_extraneous_columns(sample_df)
        assert len(result.columns) == 0
        assert len(result) == len(sample_df)  # Same number of rows

    def test_regex_no_matches(self, sample_df: pd.DataFrame) -> None:
        """Should return empty when regex matches no columns."""
        result = remove_extraneous_columns(sample_df, regex=r"^nonexistent_")
        assert len(result.columns) == 0

    def test_column_list_with_nonexistent_column(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Should only keep existing columns from list."""
        result = remove_extraneous_columns(
            sample_df, column_list=["col_a", "nonexistent"]
        )
        assert set(result.columns) == {"col_a"}

    def test_empty_dataframe(self, mock_schema: MagicMock) -> None:
        """Should handle empty DataFrame."""
        df = pd.DataFrame()
        result = remove_extraneous_columns(df, schema=mock_schema)
        assert len(result) == 0

    def test_regex_partial_match(self, sample_df: pd.DataFrame) -> None:
        """Should use regex match (not search) - only matches from start."""
        # Regex starts with ^, so only matches from beginning of string
        result = remove_extraneous_columns(sample_df, regex=r"^col_")
        assert set(result.columns) == {"col_a", "col_b", "col_c"}

    def test_returns_new_dataframe(self, sample_df: pd.DataFrame) -> None:
        """Should return new DataFrame, not modify original."""
        original_columns = sample_df.columns.tolist()
        result = remove_extraneous_columns(sample_df, column_list=["col_a"])
        assert sample_df.columns.tolist() == original_columns
        assert id(result) != id(sample_df)

    def test_preserves_data_in_kept_columns(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Should preserve data in filtered columns."""
        result = remove_extraneous_columns(
            sample_df, column_list=["col_a", "col_b"]
        )
        pd.testing.assert_series_equal(result["col_a"], sample_df["col_a"])
        pd.testing.assert_series_equal(result["col_b"], sample_df["col_b"])

    def test_regex_with_complex_pattern(self, sample_df: pd.DataFrame) -> None:
        """Should handle complex regex patterns."""
        result = remove_extraneous_columns(
            sample_df, regex=r"^(col_a|other_x)$"
        )
        assert set(result.columns) == {"col_a", "other_x"}
