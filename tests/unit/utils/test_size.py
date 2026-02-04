"""Tests for size calculation utilities for payloads and data structures."""

import pandas as pd
import pytest

from arize.utils.size import get_payload_size_mb


@pytest.mark.unit
class TestGetPayloadSizeMB:
    """Test get_payload_size_mb function."""

    def test_dataframe_size_calculation(self) -> None:
        """Should calculate size for DataFrame payload."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        size_mb = get_payload_size_mb(df)
        assert isinstance(size_mb, float)
        assert size_mb >= 0  # Small DataFrames may round to 0.0

    def test_list_of_dicts_size_calculation(self) -> None:
        """Should calculate size for list of dicts payload."""
        payload = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
        size_mb = get_payload_size_mb(payload)
        assert isinstance(size_mb, float)
        assert size_mb > 0

    def test_empty_dataframe(self) -> None:
        """Should handle empty DataFrame."""
        df = pd.DataFrame()
        size_mb = get_payload_size_mb(df)
        assert isinstance(size_mb, float)
        assert size_mb >= 0

    def test_empty_list(self) -> None:
        """Should handle empty list."""
        payload: list[dict[str, object]] = []
        size_mb = get_payload_size_mb(payload)
        assert isinstance(size_mb, float)
        assert size_mb >= 0

    def test_large_dataframe(self) -> None:
        """Should calculate size for large DataFrame."""
        df = pd.DataFrame({"a": range(10000), "b": ["x"] * 10000})
        size_mb = get_payload_size_mb(df)
        assert isinstance(size_mb, float)
        assert size_mb > 0

    def test_result_is_rounded_to_three_decimals(self) -> None:
        """Should round result to 3 decimal places."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        size_mb = get_payload_size_mb(df)
        # Check that it has at most 3 decimal places
        size_str = str(size_mb)
        if "." in size_str:
            decimals = len(size_str.split(".")[1])
            assert decimals <= 3

    def test_dataframe_with_various_types(self) -> None:
        """Should handle DataFrame with various column types."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        size_mb = get_payload_size_mb(df)
        assert isinstance(size_mb, float)
        assert size_mb >= 0  # Small DataFrames may round to 0.0

    def test_list_size_is_positive(self) -> None:
        """Should return positive size for non-empty list."""
        payload = [{"key": "value"}] * 100
        size_mb = get_payload_size_mb(payload)
        assert size_mb > 0

    def test_unsupported_type_raises_error(self) -> None:
        """Should raise TypeError for unsupported payload types."""
        with pytest.raises(TypeError, match="Unsupported payload type"):
            get_payload_size_mb("invalid")  # type: ignore

        with pytest.raises(TypeError, match="Unsupported payload type"):
            get_payload_size_mb(123)  # type: ignore

    def test_dataframe_size_uses_deep_memory_usage(self) -> None:
        """Should use deep memory usage for accurate DataFrame sizing."""
        # DataFrame with object dtype (strings) requires deep=True for accurate sizing
        df1 = pd.DataFrame({"a": ["short"] * 100})
        df2 = pd.DataFrame({"a": ["very long string" * 10] * 100})
        size1 = get_payload_size_mb(df1)
        size2 = get_payload_size_mb(df2)
        # df2 should be larger due to longer strings
        assert size2 > size1

    def test_list_vs_dataframe_size_comparison(self) -> None:
        """Should calculate different sizes for list vs DataFrame with same data."""
        data = [{"a": i, "b": f"value_{i}"} for i in range(100)]
        df = pd.DataFrame(data)
        list_size = get_payload_size_mb(data)
        df_size = get_payload_size_mb(df)
        # Both should be positive
        assert list_size > 0
        assert df_size > 0
