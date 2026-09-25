"""Unit tests for src/arize/experiments/upload.py."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arize.experiments.evaluators.types import EvaluationResultFieldNames
from arize.experiments.types import ExperimentTaskFieldNames
from arize.experiments.upload import (
    flight_schema,
    iter_flight_batches,
    source_type,
)
from arize.utils.file_sources import open_source, unified_source_schema

if TYPE_CHECKING:
    from pathlib import Path

TASK = ExperimentTaskFieldNames(example_id="my_id", output="my_out")
EVALS = {
    "quality": EvaluationResultFieldNames(
        score="score_col", metadata={"why": "meta_col"}
    )
}
SOURCE = pa.schema(
    [
        ("my_id", pa.string()),
        ("my_out", pa.struct([("a", pa.int64())])),
        ("score_col", pa.float64()),
        ("meta_col", pa.struct([("k", pa.int64())])),
        ("extra", pa.int64()),
    ]
)


def stream(paths: list[Path], batch_rows: int = 2) -> pa.Table:
    sources = [open_source(p) for p in paths]
    source_schema = unified_source_schema(sources, source_type)
    schema = flight_schema(source_schema, TASK, EVALS)
    batches = list(
        iter_flight_batches(
            sources, source_schema, schema, batch_rows, TASK, EVALS
        )
    )
    assert all(x.schema.equals(schema) for x in batches)
    return pa.Table.from_batches(batches)


def runs_table(ids: list[str], extra: bool = True) -> pa.Table:
    cols = {
        "my_id": ids,
        "my_out": [{"a": i} for i in range(len(ids))],
        "score_col": [0.5] * len(ids),
        "meta_col": [{"k": i} for i in range(len(ids))],
    }
    if extra:
        cols["extra"] = list(range(len(ids)))
    return pa.table(cols)


@pytest.mark.unit
class TestFlightSchema:
    def test_maps_task_and_evaluator_columns(self) -> None:
        schema = flight_schema(SOURCE, TASK, EVALS)
        assert set(schema.names) == {
            "example_id",
            "output",
            "eval.quality.score",
            "eval.quality.metadata.why",
            "extra",
        }
        assert schema.field("example_id").type == pa.string()
        assert schema.field("output").type == pa.string()
        assert schema.field("eval.quality.score").type == pa.float64()
        assert schema.field("eval.quality.metadata.why").type == pa.string()
        assert schema.field("extra").type == pa.int64()

    def test_scalar_metadata_and_string_output_keep_types(self) -> None:
        source = pa.schema(
            [
                ("my_id", pa.int64()),
                ("my_out", pa.large_string()),
                ("s", pa.float64()),
                ("m", pa.int64()),
            ]
        )
        evals = {
            "q": EvaluationResultFieldNames(score="s", metadata={"m": "m"})
        }
        normalized = pa.schema(
            [pa.field(f.name, source_type(f)) for f in source]
        )
        schema = flight_schema(normalized, TASK, evals)
        assert schema.field("example_id").type == pa.int64()
        assert schema.field("output").type == pa.string()
        assert schema.field("eval.q.score").type == pa.float64()
        assert schema.field("eval.q.metadata.m").type == pa.int64()

    def test_missing_task_column_raises_before_reading(self) -> None:
        with pytest.raises(ValueError, match="Missing required columns"):
            flight_schema(pa.schema([("my_id", pa.string())]), TASK, None)

    def test_missing_metadata_column_raises(self) -> None:
        evals = {
            "q": EvaluationResultFieldNames(
                score="score_col", metadata={"w": "nope"}
            )
        }
        with pytest.raises(ValueError, match="metadata column nope"):
            flight_schema(SOURCE, TASK, evals)


@pytest.mark.unit
class TestIterFlightBatches:
    def test_batches_are_transformed_and_share_schema(
        self, tmp_path: Path
    ) -> None:
        a = tmp_path / "a.parquet"
        b = tmp_path / "b.parquet"
        pq.write_table(runs_table(["e1", "e2", "e3"]), a)
        pq.write_table(runs_table(["e4"], extra=False), b)
        table = stream([a, b])
        assert table.num_rows == 4
        assert table.column("example_id").to_pylist() == [
            "e1",
            "e2",
            "e3",
            "e4",
        ]
        assert table.column("output").to_pylist() == [
            json.dumps({"a": i}) for i in (0, 1, 2, 0)
        ]
        assert table.column("eval.quality.metadata.why").to_pylist() == [
            str({"k": i}) for i in (0, 1, 2, 0)
        ]
        assert table.column("extra").to_pylist() == [0, 1, 2, None]
        assert "my_id" not in table.schema.names

    def test_file_missing_task_column_yields_nulls_without_raising(
        self, tmp_path: Path
    ) -> None:
        a = tmp_path / "a.parquet"
        b = tmp_path / "b.parquet"
        pq.write_table(runs_table(["e1"]), a)
        pq.write_table(
            pa.table(
                {
                    "my_id": ["e2"],
                    "score_col": [0.5],
                    "meta_col": [{"k": 9}],
                    "extra": [1],
                }
            ),
            b,
        )
        table = stream([a, b])
        assert table.column("output").to_pylist() == [
            json.dumps({"a": 0}),
            None,
        ]

    def test_named_pandas_index_kept_as_column(self, tmp_path: Path) -> None:
        df = runs_table(["e1", "e2"]).to_pandas()
        df.index = pd.Index(["r1", "r2"], name="row")
        f = tmp_path / "idx.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=True), f)
        table = stream([f])
        assert table.column("row").to_pylist() == ["r1", "r2"]

    def test_map_output_becomes_json_string(self, tmp_path: Path) -> None:
        f = tmp_path / "m.parquet"
        pq.write_table(
            pa.table(
                {
                    "my_id": ["e1"],
                    "my_out": pa.array(
                        [[("k", 1)]], pa.map_(pa.string(), pa.int64())
                    ),
                    "score_col": [0.5],
                    "meta_col": [{"k": 1}],
                }
            ),
            f,
        )
        table = stream([f])
        assert table.column("output").to_pylist() == [json.dumps({"k": 1})]
