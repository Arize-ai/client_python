"""Unit tests for src/arize/utils/file_sources.py."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arize.utils import file_sources
from arize.utils.file_sources import (
    conform_to_schema,
    is_path_input,
    json_encode_maps,
    open_source,
    resolve_files,
    split_oversized,
    unified_source_schema,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def identity(field: pa.Field) -> pa.DataType:
    return field.type


@pytest.mark.unit
class TestIsPathInput:
    def test_paths(self, tmp_path: Path) -> None:
        assert is_path_input("x.parquet")
        assert is_path_input(tmp_path)
        assert is_path_input(["a.parquet", tmp_path])
        assert is_path_input(("a.parquet",))

    def test_non_paths(self) -> None:
        assert not is_path_input([{"a": 1}])
        assert not is_path_input([])
        assert not is_path_input(None)

    def test_mixed_list_raises(self) -> None:
        with pytest.raises(TypeError, match="only dicts or only file paths"):
            is_path_input([{"a": 1}, "x.parquet"])


@pytest.mark.unit
class TestResolveFiles:
    def test_single_file_str_and_path(self, tmp_path: Path) -> None:
        f = write_parquet(tmp_path / "x.parquet", rows(1))
        assert resolve_files(str(f)) == [f]
        assert resolve_files(f) == [f]

    def test_directory_is_recursive_sorted_and_filtered(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        p = write_parquet(tmp_path / "a" / "x.parquet", rows(1))
        a = write_arrow(tmp_path / "b" / "y.arrow", rows(1))
        f = write_arrow(tmp_path / "z.feather", rows(1))
        (tmp_path / "ignore.csv").write_text("a\n1\n")
        assert resolve_files(tmp_path) == [p, a, f]

    def test_list_mixing_file_and_directory(self, tmp_path: Path) -> None:
        (tmp_path / "d").mkdir()
        inner = write_parquet(tmp_path / "d" / "x.parquet", rows(1))
        outer = write_parquet(tmp_path / "y.parquet", rows(1))
        assert resolve_files([outer, str(tmp_path / "d")]) == [outer, inner]

    def test_unsupported_suffix(self, tmp_path: Path) -> None:
        csv = tmp_path / "x.csv"
        csv.write_text("a\n1\n")
        with pytest.raises(ValueError, match="unsupported file type"):
            resolve_files(csv)

    def test_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_files(tmp_path / "nope.parquet")

    def test_empty_directory_and_empty_list(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no Parquet or Arrow files"):
            resolve_files(tmp_path)
        with pytest.raises(ValueError, match="no file paths"):
            resolve_files([])


@pytest.mark.unit
class TestSources:
    def test_parquet_and_arrow_sources_agree(self, tmp_path: Path) -> None:
        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", None]})
        parquet = open_source(write_parquet(tmp_path / "t.parquet", table))
        arrow = open_source(write_arrow(tmp_path / "t.feather", table))
        assert parquet.schema.equals(arrow.schema)
        assert parquet.num_rows == arrow.num_rows == 3
        assert pa.Table.from_batches(list(parquet.iter_batches(2))).equals(
            pa.Table.from_batches(list(arrow.iter_batches(2)))
        )
        assert [b.num_rows for b in arrow.iter_batches(2)] == [2, 1]
        first = next(iter(arrow.iter_batches(2, columns=["b"])))
        assert first.schema.names == ["b"]

    def test_parquet_row_groups_are_read_one_at_a_time(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "x.parquet"
        pq.write_table(rows(25), f, row_group_size=4)
        assert pq.ParquetFile(f).num_row_groups == 7
        assert [b.num_rows for b in open_source(f).iter_batches(10)] == [
            4
        ] * 6 + [1]
        assert [b.num_rows for b in open_source(f).iter_batches(3)] == (
            [3, 1] * 6 + [1]
        )

    def test_sources_hold_no_file_descriptors(self, tmp_path: Path) -> None:
        table = pa.table({"a": [1]})
        files = [
            write_parquet(tmp_path / f"p{i}.parquet", table) for i in range(25)
        ] + [write_arrow(tmp_path / f"a{i}.arrow", table) for i in range(25)]
        before = len(os.listdir("/dev/fd"))
        sources = [open_source(f) for f in files]
        assert len(os.listdir("/dev/fd")) == before
        for source in sources:
            assert sum(b.num_rows for b in source.iter_batches(10)) == 1
        assert len(os.listdir("/dev/fd")) == before


@pytest.mark.unit
class TestUnifiedSourceSchema:
    def test_missing_column_is_unioned(self, tmp_path: Path) -> None:
        a = write_parquet(
            tmp_path / "a.parquet", pa.table({"a": [1], "b": ["x"]})
        )
        b = write_parquet(tmp_path / "b.parquet", pa.table({"a": [2]}))
        schema = unified_source_schema(
            [open_source(a), open_source(b)], identity
        )
        assert schema.names == ["a", "b"]

    def test_null_column_takes_typed_column(self, tmp_path: Path) -> None:
        a = write_parquet(
            tmp_path / "a.parquet",
            pa.table({"b": pa.array([None], pa.null())}),
        )
        b = write_parquet(tmp_path / "b.parquet", pa.table({"b": ["x"]}))
        schema = unified_source_schema(
            [open_source(a), open_source(b)], identity
        )
        assert schema.field("b").type == pa.string()

    def test_widens_integer_width_and_timestamp_unit(
        self, tmp_path: Path
    ) -> None:
        a = write_parquet(
            tmp_path / "a.parquet",
            pa.table(
                {
                    "n": pa.array([1], pa.int32()),
                    "ts": pa.array([1], pa.timestamp("us")),
                }
            ),
        )
        b = write_parquet(
            tmp_path / "b.parquet",
            pa.table(
                {
                    "n": pa.array([1], pa.int64()),
                    "ts": pa.array([1], pa.timestamp("ns")),
                }
            ),
        )
        schema = unified_source_schema(
            [open_source(a), open_source(b)], identity
        )
        assert schema.field("n").type == pa.int64()
        assert schema.field("ts").type == pa.timestamp("ns")

    def test_normalize_runs_before_merging(self, tmp_path: Path) -> None:
        a = write_parquet(tmp_path / "a.parquet", pa.table({"b": [True]}))
        b = write_parquet(tmp_path / "b.parquet", pa.table({"b": ["x"]}))
        schema = unified_source_schema(
            [open_source(a), open_source(b)],
            lambda f: pa.string() if pa.types.is_boolean(f.type) else f.type,
        )
        assert schema.field("b").type == pa.string()

    def test_index_columns_dropped(self, tmp_path: Path) -> None:
        a = write_parquet(
            tmp_path / "a.parquet",
            pa.table({"a": [1], "__index_level_0__": [0]}),
        )
        assert unified_source_schema([open_source(a)], identity).names == ["a"]

    def test_type_clash_names_the_file(self, tmp_path: Path) -> None:
        a = write_parquet(tmp_path / "a.parquet", pa.table({"b": [1]}))
        b = write_parquet(tmp_path / "b.parquet", pa.table({"b": ["x"]}))
        with pytest.raises(ValueError, match=r"b\.parquet"):
            unified_source_schema([open_source(a), open_source(b)], identity)


@pytest.mark.unit
class TestJsonEncodeMaps:
    def test_maps_become_json_strings(self) -> None:
        batch = pa.record_batch(
            {
                "m": pa.array(
                    [[("k", 1), ("j", 2)], None],
                    pa.map_(pa.string(), pa.int64()),
                ),
                "n": [1, 2],
            }
        )
        out = json_encode_maps(batch)
        assert out.schema.field("m").type == pa.string()
        assert out.column("m").to_pylist() == [
            json.dumps({"k": 1, "j": 2}),
            None,
        ]
        assert out.column("n").to_pylist() == [1, 2]

    def test_no_maps_returns_same_batch(self) -> None:
        batch = pa.record_batch({"n": [1]})
        assert json_encode_maps(batch) is batch


@pytest.mark.unit
class TestConformToSchema:
    def test_fills_reorders_and_drops(self) -> None:
        batch = pa.record_batch({"b": ["x"], "extra": [1]})
        schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
        out = conform_to_schema(batch, schema)
        assert out.schema.equals(schema)
        assert out.to_pydict() == {"a": [None], "b": ["x"]}

    def test_cast_failure_names_column(self) -> None:
        batch = pa.record_batch({"n": ["a"]})
        schema = pa.schema([pa.field("n", pa.int64())])
        with pytest.raises(pa.ArrowInvalid, match="column 'n'"):
            conform_to_schema(batch, schema)


@pytest.mark.unit
class TestSplitOversized:
    def test_splits_preserving_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(file_sources, "FLIGHT_SERVER_MAX_MESSAGE_BYTES", 64)
        batch = pa.record_batch({"a": list(range(8))})
        parts = list(split_oversized(batch))
        assert len(parts) > 1
        assert all(p.nbytes <= 32 for p in parts)
        assert pa.Table.from_batches(parts).column("a").to_pylist() == list(
            range(8)
        )

    def test_single_row_over_budget_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(file_sources, "FLIGHT_SERVER_MAX_MESSAGE_BYTES", 2)
        with pytest.raises(ValueError, match="single row"):
            list(split_oversized(pa.record_batch({"a": [1]})))

    def test_small_batch_passes_through(self) -> None:
        batch = pa.record_batch({"a": [1]})
        assert list(split_oversized(batch)) == [batch]
