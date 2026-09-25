"""Shape dataset examples into Arrow record batches for Flight upload."""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from arize.datasets import errors as err
from arize.utils.file_sources import (
    conform_to_schema,
    json_encode_maps,
    split_oversized,
)
from arize.utils.openinference_conversion import (
    _should_convert_json,
    convert_boolean_columns_to_str,
    convert_datetime_columns_to_int,
    convert_default_columns_to_json_str,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from arize.utils.file_sources import FileSource

_REQUIRED_COLUMNS = (
    ("id", pa.string()),
    ("created_at", pa.int64()),
    ("updated_at", pa.int64()),
)


def prepare_examples_df(df: pd.DataFrame, current_time: int) -> pd.DataFrame:
    """Apply the column conversions every dataset upload needs.

    Datetimes become int64 milliseconds, booleans become strings, missing
    ``id``/``created_at``/``updated_at`` are filled, and dict values in JSON
    columns are serialized. Mutates and returns ``df``.
    """
    df = convert_datetime_columns_to_int(df)
    df = convert_boolean_columns_to_str(df)
    df = _set_default_columns_for_dataset(df, current_time)
    return convert_default_columns_to_json_str(df)


def _set_default_columns_for_dataset(
    df: pd.DataFrame, current_time: int
) -> pd.DataFrame:
    for col in ("created_at", "updated_at"):
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(current_time)
        else:
            df[col] = current_time

    if "id" in df.columns:
        if df["id"].isnull().any():
            df["id"] = df["id"].apply(
                lambda x: str(uuid.uuid4()) if pd.isnull(x) else x
            )
    else:
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    return df


def source_type(field: pa.Field) -> pa.DataType:
    """Map a source column type to how it is read, before file schemas merge.

    Mirrors what :func:`prepare_examples_df` does to a DataFrame, plus the
    Arrow-side conversions in :func:`iter_flight_batches`. Null and binary
    columns pass through: a null column must still merge with a typed
    column from another file, and binary columns are reported together by
    :func:`flight_schema`.
    """
    t = field.type
    if (
        field.name == "id"
        or pa.types.is_boolean(t)
        or pa.types.is_large_string(t)
        or pa.types.is_map(t)
    ):
        return pa.string()
    if pa.types.is_timestamp(t):
        return pa.int64()
    if pa.types.is_struct(t) and _should_convert_json(field.name):
        return pa.string()
    return t


def flight_type(field: pa.Field) -> pa.DataType:
    """Map a merged column type to the type it is uploaded as.

    Raises:
        BinaryColumnError: If the column holds bytes.
    """
    t = field.type
    if _is_binary(t):
        raise err.BinaryColumnError([field.name])
    if (
        field.name == "id"
        or pa.types.is_boolean(t)
        or pa.types.is_null(t)
        or pa.types.is_large_string(t)
        or pa.types.is_map(t)
    ):
        return pa.string()
    if pa.types.is_timestamp(t):
        return pa.int64()
    if pa.types.is_struct(t) and _should_convert_json(field.name):
        return pa.string()
    return t


def _is_binary(t: pa.DataType) -> bool:
    return (
        pa.types.is_binary(t)
        or pa.types.is_large_binary(t)
        or pa.types.is_fixed_size_binary(t)
    )


def flight_schema(source: pa.Schema) -> pa.Schema:
    """Append the required columns to a schema already mapped by :func:`flight_type`.

    Raises:
        BinaryColumnError: If any column holds bytes.
    """
    binary = [f.name for f in source if _is_binary(f.type)]
    if binary:
        raise err.BinaryColumnError(binary)
    fields = [pa.field(f.name, flight_type(f)) for f in source]
    names = {f.name for f in fields}
    fields.extend(
        pa.field(name, dtype)
        for name, dtype in _REQUIRED_COLUMNS
        if name not in names
    )
    return pa.schema(fields)


def check_unique_ids(sources: Sequence[FileSource], batch_rows: int) -> None:
    """Read only the ``id`` column and fail if any value repeats.

    Ids are reduced to 16-byte digests in a pre-sized array and compared
    after one sort, so the check costs 16 bytes per row rather than a
    Python string per row. Null ids are skipped because they are replaced
    with fresh UUIDs.

    Raises:
        IDColumnUniqueConstraintError: If an id appears more than once.
    """
    digests = np.empty(sum(s.num_rows for s in sources), dtype="S16")
    count = 0
    for source in sources:
        if "id" not in source.schema.names:
            continue
        for batch in source.iter_batches(batch_rows, columns=["id"]):
            ids = _id_strings(batch.column(0)).drop_null().to_pylist()
            digests[count : count + len(ids)] = [
                hashlib.blake2b(id_.encode(), digest_size=16).digest()
                for id_ in ids
            ]
            count += len(ids)
    used = digests[:count]
    used.sort()
    if count > 1 and bool(np.any(used[1:] == used[:-1])):
        raise err.IDColumnUniqueConstraintError()


def iter_flight_batches(
    sources: Sequence[FileSource],
    schema: pa.Schema,
    batch_rows: int,
    current_time: int,
) -> Iterator[pa.RecordBatch]:
    """Yield converted batches that all carry exactly ``schema``."""
    for source in sources:
        for batch in source.iter_batches(batch_rows):
            if batch.num_rows == 0:
                continue
            yield from split_oversized(_conform(batch, schema, current_time))


def _conform(
    batch: pa.RecordBatch, schema: pa.Schema, current_time: int
) -> pa.RecordBatch:
    batch = _cast_arrow_side(json_encode_maps(batch))
    df = prepare_examples_df(batch.to_pandas(), current_time)
    converted = pa.RecordBatch.from_pandas(df, preserve_index=False)
    return conform_to_schema(converted, schema)


def _cast_arrow_side(batch: pa.RecordBatch) -> pa.RecordBatch:
    # Timestamps of any unit become epoch milliseconds here because the
    # pandas helper only recognizes nanosecond columns; booleans become
    # "True"/"False" here because a bool column with nulls reaches pandas as
    # object dtype and the pandas helper skips it. The id cast keeps an
    # integer id column with nulls from turning into floats in pandas.
    columns = []
    for field, column in zip(batch.schema, batch.columns, strict=True):
        if pa.types.is_timestamp(field.type):
            column = column.cast(
                pa.timestamp("ms", tz=field.type.tz), safe=False
            ).cast(pa.int64())
        elif pa.types.is_boolean(field.type):
            column = pc.if_else(column, "True", "False")
        elif field.name == "id":
            column = _id_strings(column)
        columns.append(column)
    return pa.RecordBatch.from_arrays(columns, names=batch.schema.names)


def _id_strings(column: pa.Array) -> pa.Array:
    # A float NaN would otherwise cast to the literal id "nan"; pandas treats
    # it as missing, so it must become null here too.
    if pa.types.is_floating(column.type):
        column = pc.if_else(
            pc.is_nan(column), pa.scalar(None, column.type), column
        )
    return column.cast(pa.string())
