"""Shape experiment runs into Arrow record batches for Flight upload."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from arize.experiments.functions import transform_to_experiment_format
from arize.utils.file_sources import (
    conform_to_schema,
    json_encode_maps,
    split_oversized,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from arize.experiments.evaluators.types import EvaluationResultFieldNames
    from arize.experiments.types import ExperimentTaskFieldNames
    from arize.utils.file_sources import FileSource


def source_type(field: pa.Field) -> pa.DataType:
    """Map a source column type to how it is read: maps become JSON strings."""
    t = field.type
    if pa.types.is_large_string(t) or pa.types.is_map(t):
        return pa.string()
    return t


def flight_schema(
    source: pa.Schema,
    task_fields: ExperimentTaskFieldNames,
    evaluator_columns: dict[str, EvaluationResultFieldNames] | None,
) -> pa.Schema:
    """Derive the schema every uploaded batch is cast to.

    Runs the run-format transform over a one-row probe whose cells name
    their own column, so the output tells which source column each
    canonical column comes from and the transform's own required-column
    validation runs before any data is read.

    Raises:
        ValueError: If a column named by ``task_fields`` is missing.
    """
    probe = transform_to_experiment_format(
        pd.DataFrame({f.name: [f.name] for f in source}),
        task_fields,
        evaluator_columns,
    )
    fields = []
    for target in probe.columns:
        # Holds only while the transform renames and never derives values.
        origin = probe[target].iloc[0]
        if origin not in source.names:
            raise ValueError(
                f"cannot trace column {target!r} back to a source column"
            )
        field = source.field(origin)
        fields.append(pa.field(target, _flight_type(target, field.type)))
    return pa.schema(fields)


def _flight_type(name: str, t: pa.DataType) -> pa.DataType:
    if name == "output" and pa.types.is_struct(t):
        return pa.string()
    if (
        name.startswith("eval.")
        and ".metadata." in name
        and not (
            pa.types.is_integer(t)
            or pa.types.is_floating(t)
            or pa.types.is_string(t)
            or pa.types.is_boolean(t)
            or pa.types.is_null(t)
        )
    ):
        return pa.string()
    return t


def iter_flight_batches(
    sources: Sequence[FileSource],
    source_schema: pa.Schema,
    schema: pa.Schema,
    batch_rows: int,
    task_fields: ExperimentTaskFieldNames,
    evaluator_columns: dict[str, EvaluationResultFieldNames] | None,
) -> Iterator[pa.RecordBatch]:
    """Yield run batches that all carry exactly ``schema``.

    Each batch is first conformed to ``source_schema`` so a file that lacks
    a column still passes the transform's required-column check.
    """
    for source in sources:
        for batch in source.iter_batches(batch_rows):
            if batch.num_rows == 0:
                continue
            batch = conform_to_schema(json_encode_maps(batch), source_schema)
            df = transform_to_experiment_format(
                batch.to_pandas(), task_fields, evaluator_columns
            )
            converted = pa.RecordBatch.from_pandas(df, preserve_index=False)
            yield from split_oversized(conform_to_schema(converted, schema))
