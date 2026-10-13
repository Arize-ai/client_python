"""Tests for JSON record normalization helpers."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from arize.utils.records import normalize_json_value, to_json_records


@pytest.mark.unit
class TestNormalizeJsonValue:
    """Tests for normalize_json_value."""

    def test_converts_ndarray_to_list(self) -> None:
        """Arrays become plain lists of Python scalars."""
        out = normalize_json_value(np.array([1, 2, 3], dtype=np.int64))
        assert out == [1, 2, 3]
        assert all(type(x) is int for x in out)  # type: ignore[union-attr]

    def test_converts_nested_containers(self) -> None:
        """Arrays nested inside dicts, lists and tuples are converted too."""
        out = normalize_json_value(
            {
                "a": np.array([[1.0, 2.0]]),
                "b": (np.int32(4), [np.bool_(True)]),
                "c": np.float64(2.5),
            }
        )
        assert out == {"a": [[1.0, 2.0]], "b": [4, [True]], "c": 2.5}

    def test_converts_datetime64_and_nat(self) -> None:
        """datetime64 values become ISO strings and NaT becomes None."""
        arr = np.array(["2024-01-01T00:00:00", "NaT"], dtype="datetime64[ns]")
        out = normalize_json_value(arr)
        assert out == ["2024-01-01T00:00:00.000000000", None]
        assert normalize_json_value(np.datetime64("NaT", "ns")) is None

    def test_converts_pd_na_to_none(self) -> None:
        """pd.NA is not JSON-serializable, so it becomes None."""
        assert normalize_json_value({"a": pd.NA, "b": [pd.NA]}) == {
            "a": None,
            "b": [None],
        }

    def test_converts_nan_and_inf_to_none(self) -> None:
        """NaN and Infinity are not valid JSON, so they become None."""
        out = normalize_json_value(
            {
                "a": float("nan"),
                "b": float("inf"),
                "c": float("-inf"),
                "d": np.float64("nan"),
                "e": np.array([1.5, np.inf]),
                "f": [{"score": float("nan")}, {"score": 0.9}],
            }
        )
        assert out == {
            "a": None,
            "b": None,
            "c": None,
            "d": None,
            "e": [1.5, None],
            "f": [{"score": None}, {"score": 0.9}],
        }

    def test_leaves_plain_values_alone(self) -> None:
        """Python scalars and strings pass through unchanged."""
        assert normalize_json_value("x") == "x"
        assert normalize_json_value(None) is None
        assert normalize_json_value(3) == 3
        stamp = datetime(2024, 1, 1)
        assert normalize_json_value(stamp) is stamp


@pytest.mark.unit
class TestToJsonRecords:
    """Tests for to_json_records."""

    def test_dataframe_becomes_normalized_records(self) -> None:
        """A DataFrame turns into one dict per row with plain lists."""
        df = pd.DataFrame(
            {"id": ["a", "b"], "vec": [np.array([1, 2]), np.array([3])]}
        )
        assert to_json_records(df) == [
            {"id": "a", "vec": [1, 2]},
            {"id": "b", "vec": [3]},
        ]

    def test_list_of_dicts_is_normalized_and_copied(self) -> None:
        """A list of dicts is normalized without touching the input."""
        source = [{"vec": np.array([1, 2]), "n": np.int64(7)}]
        out = to_json_records(source)
        assert out == [{"vec": [1, 2], "n": 7}]
        assert isinstance(source[0]["vec"], np.ndarray)

    def test_empty_inputs(self) -> None:
        """Empty inputs yield empty record lists."""
        assert to_json_records([]) == []
        assert to_json_records(pd.DataFrame()) == []
