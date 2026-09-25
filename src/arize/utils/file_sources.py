"""Read Parquet and Arrow IPC files as record batches for Flight upload."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeGuard

import pyarrow as pa
import pyarrow.parquet as pq

from arize.constants.pyarrow import FLIGHT_SERVER_MAX_MESSAGE_BYTES

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

INDEX_COLUMN = re.compile(r"__index_level_\d+__")


class FileSource(Protocol):
    """A data file whose rows can be read as record batches without loading it all.

    Opening a source reads only the footer; the file is held open only
    while :meth:`iter_batches` runs.
    """

    path: Path
    schema: pa.Schema
    num_rows: int

    def iter_batches(
        self, batch_rows: int, columns: list[str] | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Yield batches of at most ``batch_rows`` rows, optionally projected."""
        ...


class _ParquetSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.schema: pa.Schema = pq.read_schema(path)
        self.num_rows: int = pq.read_metadata(path).num_rows

    def iter_batches(
        self, batch_rows: int, columns: list[str] | None = None
    ) -> Iterator[pa.RecordBatch]:
        # Iterating the whole file reads ahead across row groups, so a whole
        # file can end up resident; one row group at a time bounds memory to
        # roughly that row group.
        file = pq.ParquetFile(self.path)
        try:
            for i in range(file.num_row_groups):
                yield from file.iter_batches(
                    batch_size=batch_rows,
                    row_groups=[i],
                    columns=columns,
                    use_threads=False,
                )
        finally:
            file.close()


class _ArrowFileSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        with pa.memory_map(str(path), "r") as source:
            file = pa.ipc.open_file(source)
            self.schema: pa.Schema = file.schema
            self.num_rows: int = sum(
                file.get_batch(i).num_rows
                for i in range(file.num_record_batches)
            )

    def iter_batches(
        self, batch_rows: int, columns: list[str] | None = None
    ) -> Iterator[pa.RecordBatch]:
        # Memory-mapping keeps get_batch zero-copy, so iterating touches
        # only the pages actually read.
        with pa.memory_map(str(self.path), "r") as source:
            file = pa.ipc.open_file(source)
            for i in range(file.num_record_batches):
                batch = file.get_batch(i)
                if columns is not None:
                    batch = batch.select(columns)
                for start in range(0, batch.num_rows, batch_rows):
                    yield batch.slice(start, batch_rows)


_SOURCES: dict[str, type[_ParquetSource] | type[_ArrowFileSource]] = {
    ".parquet": _ParquetSource,
    ".arrow": _ArrowFileSource,
    ".feather": _ArrowFileSource,
}


def is_path_input(
    value: object,
) -> TypeGuard[str | os.PathLike[str] | Sequence[str | os.PathLike[str]]]:
    """Tell file-path input apart from a list of example dicts.

    Raises:
        TypeError: If a list mixes dicts and paths.
    """
    if isinstance(value, (str, os.PathLike)):
        return True
    if not isinstance(value, (list, tuple)) or not value:
        return False
    is_path = [isinstance(e, (str, os.PathLike)) for e in value]
    if all(is_path):
        return True
    if any(is_path):
        raise TypeError("list must contain only dicts or only file paths")
    return False


def resolve_files(
    examples: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> list[Path]:
    """Expand a path, directory, or list of them into data files.

    Directories are searched recursively for Parquet and Arrow IPC files.

    Raises:
        FileNotFoundError: If a path does not exist.
        ValueError: If a file has an unsupported suffix or no files are found.
    """
    inputs = (
        [examples]
        if isinstance(examples, (str, os.PathLike))
        else list(examples)
    )
    if not inputs:
        raise ValueError("no file paths given")
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found = sorted(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in _SOURCES
            )
            if not found:
                raise ValueError(
                    f"no Parquet or Arrow files found under {path}"
                )
            files.extend(found)
        elif path.is_file():
            if path.suffix.lower() not in _SOURCES:
                raise ValueError(
                    f"unsupported file type {path.suffix!r} for {path}; "
                    f"expected one of {sorted(_SOURCES)}"
                )
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return files


def open_source(path: Path) -> FileSource:
    """Read a data file's footer with the reader its suffix selects."""
    return _SOURCES[path.suffix.lower()](path)


def unified_source_schema(
    sources: Sequence[FileSource],
    normalize: Callable[[pa.Field], pa.DataType],
) -> pa.Schema:
    """Merge the file schemas after mapping each field through ``normalize``.

    Fields that still differ are widened where Arrow allows it (integer
    width, timestamp unit); anything else raises ValueError naming the file.
    """
    schemas = [
        pa.schema(
            [
                pa.field(f.name, normalize(f))
                for f in source.schema
                if not INDEX_COLUMN.fullmatch(f.name)
            ]
        )
        for source in sources
    ]
    schema = schemas[0]
    for source, other in zip(sources[1:], schemas[1:], strict=True):
        schema = _unify(schema, other, source.path)
    return schema


def _unify(a: pa.Schema, b: pa.Schema, path: Path) -> pa.Schema:
    try:
        try:
            return pa.unify_schemas([a, b], promote_options="permissive")
        except TypeError:
            return pa.unify_schemas([a, b])
    except pa.ArrowException as e:
        raise ValueError(
            f"{path} has a column type incompatible with the other files: {e}"
        ) from e


def json_encode_maps(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Serialize Arrow map columns to JSON strings.

    pandas turns a map into a list of key/value tuples, which the JSON
    column conversions do not recognize, so maps are encoded here first.
    """
    if not any(pa.types.is_map(t) for t in batch.schema.types):
        return batch
    columns = []
    for field, column in zip(batch.schema, batch.columns, strict=True):
        if pa.types.is_map(field.type):
            column = pa.array(
                [
                    None if pairs is None else json.dumps(dict(pairs))
                    for pairs in column.to_pylist()
                ],
                pa.string(),
            )
        columns.append(column)
    return pa.RecordBatch.from_arrays(columns, names=batch.schema.names)


def conform_to_schema(
    batch: pa.RecordBatch, schema: pa.Schema
) -> pa.RecordBatch:
    """Reorder, fill, and cast ``batch`` so it carries exactly ``schema``.

    Columns missing from the batch become nulls; extra columns and schema
    metadata are dropped.

    Raises:
        ArrowInvalid: If a column cannot be cast to its pinned type.
    """
    arrays = []
    for field in schema:
        if field.name not in batch.schema.names:
            arrays.append(pa.nulls(batch.num_rows, field.type))
            continue
        column = batch.column(field.name)
        try:
            arrays.append(column.cast(field.type))
        except pa.ArrowException as e:
            raise pa.ArrowInvalid(
                f"column {field.name!r}: cannot convert {column.type} "
                f"to {field.type}: {e}"
            ) from e
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def split_oversized(batch: pa.RecordBatch) -> Iterator[pa.RecordBatch]:
    """Halve ``batch`` until each piece fits the Flight server's message limit."""
    budget = FLIGHT_SERVER_MAX_MESSAGE_BYTES // 2
    if batch.nbytes <= budget:
        yield batch
        return
    if batch.num_rows == 1:
        raise ValueError(
            f"a single row is {batch.nbytes} bytes; the Flight server "
            f"accepts at most {budget} bytes per record batch"
        )
    half = batch.num_rows // 2
    yield from split_oversized(batch.slice(0, half))
    yield from split_oversized(batch.slice(half))
