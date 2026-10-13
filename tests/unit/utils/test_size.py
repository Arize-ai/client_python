"""Tests for size calculation utilities for payloads and data structures."""

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from arize.utils.records import to_json_records
from arize.utils.size import to_json_records_within

MB = 1024 * 1024


class _Unserializable:
    def __str__(self) -> str:
        raise AssertionError("row past the size limit was converted")


@pytest.mark.unit
class TestToJsonRecordsWithin:
    """Test to_json_records_within function."""

    def test_list_under_limit_returns_records(self) -> None:
        """Should return the same records as to_json_records."""
        payload = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        assert to_json_records_within(payload, 1) == to_json_records(payload)

    def test_dataframe_under_limit_returns_records(self) -> None:
        """Should convert DataFrame rows like to_json_records."""
        df = pd.DataFrame(
            {
                "n": np.array([1, 2], dtype=np.int64),
                "f": np.array([1.5, np.nan]),
                "when": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "arr": [np.array(["a", "b"]), np.array([], dtype=object)],
            }
        )
        result = to_json_records_within(df, 1)
        assert result is not None
        assert len(result) == 2
        assert result[0]["arr"] == ["a", "b"]
        assert type(result[0]["n"]) is int
        assert result[1]["arr"] == []

    def test_dataframe_nullable_dtypes_match_to_json_records(self) -> None:
        """Missing values in nullable dtypes become None, as with to_dict."""
        df = pd.DataFrame(
            {
                "i": pd.array([1, None], dtype="Int64"),
                "s": pd.array(["x", None], dtype="string"),
                "b": pd.array([True, None], dtype="boolean"),
            }
        )
        result = to_json_records_within(df, 1)
        assert result == to_json_records(df)
        assert result[1] == {"i": None, "s": None, "b": None}

    def test_empty_inputs(self) -> None:
        """Should return an empty list for empty input."""
        assert to_json_records_within([], 1) == []
        assert to_json_records_within(pd.DataFrame(), 1) == []

    def test_over_limit_returns_none(self) -> None:
        """Should return None when the JSON body exceeds the limit."""
        payload = [{"a": i, "b": "x" * 50_000} for i in range(40)]
        assert to_json_records_within(payload, 1) is None

    def test_body_exactly_at_limit_returns_records(self) -> None:
        """A body exactly at the limit still fits."""
        payload = [{"a": i, "b": "x" * 1000} for i in range(30)]
        limit_mb = len(json.dumps(payload).encode("utf-8")) / MB
        assert to_json_records_within(payload, limit_mb) == payload

    def test_body_one_byte_over_limit_returns_none(self) -> None:
        """Separators between records count toward the size."""
        payload = [{"a": i, "b": "x" * 1000} for i in range(30)]
        limit_mb = (len(json.dumps(payload).encode("utf-8")) - 1) / MB
        assert to_json_records_within(payload, limit_mb) is None

    def test_multibyte_characters_count_as_bytes(self) -> None:
        """Size is measured in UTF-8 bytes, not characters."""
        payload = [{"s": "é" * 1000}]
        ascii_body_mb = len(json.dumps(payload, ensure_ascii=False)) / MB
        limit_mb = (len(json.dumps(payload).encode("utf-8")) - 1) / MB
        assert limit_mb > ascii_body_mb
        assert to_json_records_within(payload, limit_mb) is None

    def test_list_stops_converting_after_limit(self) -> None:
        """Rows after the limit is passed are never converted."""
        payload = [{"big": "x" * (2 * MB)}, {"bad": _Unserializable()}]
        assert to_json_records_within(payload, 1) is None

    def test_dataframe_stops_converting_after_limit(self) -> None:
        """DataFrame rows after the limit is passed are never converted."""
        df = pd.DataFrame({"v": ["x" * (2 * MB), _Unserializable()]})
        assert to_json_records_within(df, 1) is None

    def test_dataframe_with_arrow_ndarray_cells_measures_serialized_size(
        self,
    ) -> None:
        """Array cells from Arrow count at their serialized size, not their header size."""
        rows = 50
        values = [float(i) for i in range(5000)]
        df = pa.table(
            {
                "embedding": pa.array(
                    [values] * rows, type=pa.list_(pa.float64())
                ),
                "label": pa.array(["a"] * rows),
            }
        ).to_pandas()
        assert isinstance(df["embedding"].iloc[0], np.ndarray)
        records = [{"embedding": values, "label": "a"} for _ in range(rows)]
        body_mb = len(json.dumps(records).encode("utf-8")) / MB
        memory_usage_mb = df.memory_usage(deep=True).sum() / MB
        assert body_mb > 5 * memory_usage_mb

        assert to_json_records_within(df, body_mb) == records
        assert to_json_records_within(df, body_mb - 1 / MB) is None

    def test_unsupported_type_raises_error(self) -> None:
        """Should raise TypeError for unsupported payload types."""
        with pytest.raises(TypeError, match="Unsupported payload type"):
            to_json_records_within("invalid", 1)  # type: ignore[arg-type]
