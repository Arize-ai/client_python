"""Unit tests for src/arize/datasets/upload.py."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arize.datasets import errors as err
from arize.datasets.upload import (
    check_unique_ids,
    flight_schema,
    iter_flight_batches,
    prepare_examples_df,
    source_type,
)
from arize.utils.file_sources import open_source, unified_source_schema

if TYPE_CHECKING:
    from pathlib import Path

NOW = 1_700_000_000_000


def write_parquet(path: Path, table: pa.Table) -> Path:
    pq.write_table(table, path)
    return path


def write_arrow(path: Path, table: pa.Table) -> Path:
    with (
        pa.OSFile(str(path), "wb") as sink,
        pa.ipc.new_file(sink, table.schema) as writer,
    ):
        writer.write_table(table)
    return path


def rows(n: int) -> pa.Table:
    return pa.table({"a": list(range(n))})


def collect(paths: list[Path], batch_rows: int = 10) -> list[pa.RecordBatch]:
    sources = [open_source(p) for p in paths]
    schema = flight_schema(unified_source_schema(sources, source_type))
    return list(iter_flight_batches(sources, schema, batch_rows, NOW))


@pytest.mark.unit
class TestFlightSchema:
    @pytest.mark.parametrize(
        ("name", "source_type", "expected"),
        [
            ("ts", pa.timestamp("us"), pa.int64()),
            ("ts", pa.timestamp("ns", tz="UTC"), pa.int64()),
            ("flag", pa.bool_(), pa.string()),
            ("empty", pa.null(), pa.string()),
            ("big", pa.large_string(), pa.string()),
            ("metadata", pa.struct([("k", pa.int64())]), pa.string()),
            (
                "other",
                pa.struct([("k", pa.int64())]),
                pa.struct([("k", pa.int64())]),
            ),
            ("id", pa.int64(), pa.string()),
            ("score", pa.float64(), pa.float64()),
            ("m", pa.map_(pa.string(), pa.int64()), pa.string()),
        ],
    )
    def test_type_mapping(
        self, name: str, source_type: pa.DataType, expected: pa.DataType
    ) -> None:
        schema = flight_schema(pa.schema([pa.field(name, source_type)]))
        assert schema.field(name).type == expected

    def test_required_columns_appended(self) -> None:
        schema = flight_schema(pa.schema([pa.field("a", pa.int64())]))
        assert schema.names == ["a", "id", "created_at", "updated_at"]
        assert schema.field("id").type == pa.string()
        assert schema.field("created_at").type == pa.int64()

    def test_existing_required_columns_kept_in_place(self) -> None:
        schema = flight_schema(
            pa.schema(
                [pa.field("created_at", pa.int64()), pa.field("a", pa.int64())]
            )
        )
        assert schema.names == ["created_at", "a", "id", "updated_at"]

    def test_binary_columns_rejected_together(self) -> None:
        with pytest.raises(err.BinaryColumnError) as excinfo:
            flight_schema(
                pa.schema(
                    [
                        pa.field("raw", pa.binary()),
                        pa.field("ok", pa.string()),
                        pa.field("big", pa.large_binary()),
                    ]
                )
            )
        assert excinfo.value.column_names == ["raw", "big"]


@pytest.mark.unit
class TestCheckUniqueIds:
    def test_unique_across_files_passes(self, tmp_path: Path) -> None:
        a = write_parquet(tmp_path / "a.parquet", pa.table({"id": ["1", "2"]}))
        b = write_parquet(tmp_path / "b.parquet", pa.table({"id": ["3"]}))
        check_unique_ids([open_source(a), open_source(b)], 1)

    def test_duplicate_across_files(self, tmp_path: Path) -> None:
        a = write_parquet(tmp_path / "a.parquet", pa.table({"id": ["1"]}))
        b = write_parquet(tmp_path / "b.parquet", pa.table({"id": ["1"]}))
        with pytest.raises(err.IDColumnUniqueConstraintError):
            check_unique_ids([open_source(a), open_source(b)], 10)

    def test_duplicate_within_file(self, tmp_path: Path) -> None:
        a = write_parquet(tmp_path / "a.parquet", pa.table({"id": ["1", "1"]}))
        with pytest.raises(err.IDColumnUniqueConstraintError):
            check_unique_ids([open_source(a)], 10)

    def test_nulls_ignored(self, tmp_path: Path) -> None:
        a = write_parquet(
            tmp_path / "a.parquet", pa.table({"id": [None, None, "1"]})
        )
        check_unique_ids([open_source(a)], 10)

    def test_int_and_string_ids_compare_as_strings(
        self, tmp_path: Path
    ) -> None:
        a = write_parquet(tmp_path / "a.parquet", pa.table({"id": [1]}))
        b = write_parquet(tmp_path / "b.parquet", pa.table({"id": ["1"]}))
        with pytest.raises(err.IDColumnUniqueConstraintError):
            check_unique_ids([open_source(a), open_source(b)], 10)

    def test_file_without_id_is_skipped(self, tmp_path: Path) -> None:
        a = write_parquet(tmp_path / "a.parquet", rows(2))
        check_unique_ids([open_source(a)], 10)


@pytest.mark.unit
class TestIterFlightBatches:
    def test_batches_share_schema_and_cover_all_rows(
        self, tmp_path: Path
    ) -> None:
        f = write_parquet(tmp_path / "x.parquet", rows(25))
        sources = [open_source(f)]
        schema = flight_schema(unified_source_schema(sources, source_type))
        batches = list(iter_flight_batches(sources, schema, 10, NOW))
        assert [b.num_rows for b in batches] == [10, 10, 5]
        assert all(b.schema.equals(schema) for b in batches)

    def test_empty_file_yields_nothing(self, tmp_path: Path) -> None:
        empty = write_parquet(
            tmp_path / "e.parquet", pa.table({"a": pa.array([], pa.int64())})
        )
        full = write_parquet(tmp_path / "f.parquet", rows(3))
        assert sum(b.num_rows for b in collect([empty, full])) == 3

    def test_missing_column_filled_with_nulls(self, tmp_path: Path) -> None:
        a = write_parquet(
            tmp_path / "a.parquet", pa.table({"a": [1], "b": ["x"]})
        )
        b = write_parquet(tmp_path / "b.parquet", pa.table({"a": [2]}))
        table = pa.Table.from_batches(collect([a, b]))
        assert table.column("b").to_pylist() == ["x", None]

    def test_defaults_filled(self, tmp_path: Path) -> None:
        f = write_parquet(tmp_path / "x.parquet", rows(3))
        table = pa.Table.from_batches(collect([f]))
        ids = table.column("id").to_pylist()
        assert len(set(ids)) == 3
        assert all(len(i) == 36 for i in ids)
        assert table.column("created_at").to_pylist() == [NOW] * 3
        assert table.column("updated_at").to_pylist() == [NOW] * 3

    def test_value_conversions(self, tmp_path: Path) -> None:
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        f = write_parquet(
            tmp_path / "x.parquet",
            pa.table(
                {
                    "flag": [True, False],
                    "ts": pa.array(
                        [ts, ts.replace(microsecond=123_456)],
                        pa.timestamp("us", tz="UTC"),
                    ),
                    "n": pa.array([1, None], pa.int64()),
                    "metadata": [{"k": 1}, {"k": 2}],
                }
            ),
        )
        table = pa.Table.from_batches(collect([f]))
        assert table.column("flag").to_pylist() == ["True", "False"]
        assert table.column("ts").to_pylist() == [
            1_704_067_200_000,
            1_704_067_200_123,
        ]
        assert table.column("n").type == pa.int64()
        assert table.column("n").to_pylist() == [1, None]
        assert table.column("metadata").to_pylist() == [
            json.dumps({"k": 1}),
            json.dumps({"k": 2}),
        ]

    def test_null_column_merges_with_typed_column(self, tmp_path: Path) -> None:
        a = write_parquet(
            tmp_path / "a.parquet", pa.table({"n": pa.array([None], pa.null())})
        )
        b = write_parquet(
            tmp_path / "b.parquet", pa.table({"n": pa.array([7], pa.int64())})
        )
        table = pa.Table.from_batches(collect([a, b]))
        assert table.column("n").type == pa.int64()
        assert table.column("n").to_pylist() == [None, 7]

    def test_nan_float_ids_become_fresh_uuids(self, tmp_path: Path) -> None:
        f = write_parquet(
            tmp_path / "ids.parquet",
            pa.table(
                {
                    "id": pa.array([1.0, float("nan"), float("nan")]),
                    "x": [1, 2, 3],
                }
            ),
        )
        check_unique_ids([open_source(f)], 10)
        ids = pa.Table.from_batches(collect([f])).column("id").to_pylist()
        assert ids[0] == "1"
        assert len(ids[1]) == len(ids[2]) == 36
        assert ids[1] != ids[2]

    def test_nullable_bools_are_consistent_across_batches(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "bools.parquet"
        with pq.ParquetWriter(f, pa.schema([("flag", pa.bool_())])) as w:
            w.write_table(pa.table({"flag": pa.array([True, False])}))
            w.write_table(
                pa.table({"flag": pa.array([True, None], pa.bool_())})
            )
        table = pa.Table.from_batches(collect([f]))
        assert table.column("flag").to_pylist() == [
            "True",
            "False",
            "True",
            None,
        ]

    def test_map_columns_become_json_strings(self, tmp_path: Path) -> None:
        maps = pa.array([[("k", 1)], None], pa.map_(pa.string(), pa.int64()))
        f = write_parquet(
            tmp_path / "m.parquet", pa.table({"metadata": maps, "other": maps})
        )
        table = pa.Table.from_batches(collect([f]))
        for name in ("metadata", "other"):
            assert table.column(name).to_pylist() == [
                json.dumps({"k": 1}),
                None,
            ]

    def test_named_pandas_index_kept_as_column(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {"a": [1, 2]}, index=pd.Index(["r1", "r2"], name="row")
        )
        f = write_parquet(
            tmp_path / "x.parquet",
            pa.Table.from_pandas(df, preserve_index=True),
        )
        table = pa.Table.from_batches(collect([f]))
        assert table.column("row").to_pylist() == ["r1", "r2"]

    def test_struct_column_outside_json_names_round_trips(
        self, tmp_path: Path
    ) -> None:
        f = write_parquet(
            tmp_path / "x.parquet",
            pa.table({"custom": [{"k": 1, "s": "a"}, {"k": None, "s": None}]}),
        )
        table = pa.Table.from_batches(collect([f]))
        assert table.column("custom").to_pylist() == [
            {"k": 1, "s": "a"},
            {"k": None, "s": None},
        ]
        assert pa.types.is_struct(table.column("custom").type)

    def test_pandas_index_column_dropped(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2]}, index=[5, 6])
        f = write_parquet(
            tmp_path / "x.parquet",
            pa.Table.from_pandas(df, preserve_index=True),
        )
        assert "__index_level_0__" in open_source(f).schema.names
        batches = collect([f])
        assert "__index_level_0__" not in batches[0].schema.names

    def test_arrow_file_input(self, tmp_path: Path) -> None:
        f = write_arrow(tmp_path / "x.arrow", pa.table({"a": [1, 2, 3]}))
        table = pa.Table.from_batches(collect([f], batch_rows=2))
        assert table.column("a").to_pylist() == [1, 2, 3]
        assert table.schema.names == ["a", "id", "created_at", "updated_at"]


@pytest.mark.unit
class TestPrepareExamplesDf:
    def test_matches_legacy_conversions(self) -> None:
        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2024-01-01T00:00:00Z"]),
                "flag": [True],
                "metadata": [{"k": 1}],
            }
        )
        out = prepare_examples_df(df, NOW)
        assert out["ts"].tolist() == [1_704_067_200_000]
        assert out["flag"].tolist() == ["True"]
        assert out["metadata"].tolist() == [json.dumps({"k": 1})]
        assert out["created_at"].tolist() == [NOW]
        assert out["updated_at"].tolist() == [NOW]
        assert len(out["id"].iloc[0]) == 36

    def test_fills_only_null_defaults(self) -> None:
        df = pd.DataFrame({"id": ["keep", None], "created_at": [1.0, None]})
        out = prepare_examples_df(df, NOW)
        assert out["id"].iloc[0] == "keep"
        assert len(out["id"].iloc[1]) == 36
        assert out["created_at"].tolist() == [1.0, NOW]
