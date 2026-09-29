"""Apache Arrow utilities for data serialization and file operations."""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from arize.constants.pyarrow import (
    DEFAULT_FLIGHT_BATCH_BUDGET_BYTES,
    FLIGHT_SERVER_MAX_MESSAGE_BYTES,
)
from arize.exceptions.auth import AuthenticationError
from arize.exceptions.http import APIError
from arize.logging import get_arize_project_url, log_a_list

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import requests

    from arize._generated.protocol.rec import public_pb2 as pb2

logger = logging.getLogger(__name__)


def post_arrow_table(
    files_url: str,
    pa_table: pa.Table,
    proto_schema: pb2.Schema,
    headers: dict[str, str],
    timeout: float | None,
    verify: bool,
    max_chunksize: int,
    tmp_dir: str = "",
) -> requests.Response:
    """Post a PyArrow table to Arize via HTTP file upload.

    Args:
        files_url: The URL endpoint for file uploads.
        pa_table: The PyArrow table containing the data.
        proto_schema: The protobuf schema for the data.
        headers: HTTP headers for the request.
        timeout: Request timeout in seconds, or :obj:`None` for no timeout.
        verify: Whether to verify SSL certificates.
        max_chunksize: Maximum chunk size for splitting large tables.
        tmp_dir: Temporary directory for serialization. Defaults to "".

    Returns:
        The HTTP response from the upload request.
    """
    # We import here to avoid depending on requests for all arrow utils
    import requests

    logger.debug(
        "Preparing to log Arrow table via file upload",
        extra={"rows": pa_table.num_rows, "cols": pa_table.num_columns},
    )

    logger.debug("Serializing schema")
    base64_schema = base64.b64encode(proto_schema.SerializeToString())
    pa_schema = _append_to_pyarrow_metadata(
        pa_table.schema, {"arize-schema": base64_schema}
    )

    # --- decide output file path ---
    # cases:
    # 1) tmp_dir == ""        -> we own a TemporaryDirectory, we write to a file
    #                            in it, clean the entire dir
    # 2) tmp_dir is a dir     -> user owns the directory, we create a temp file
    #                            inside it (and remove only that file)
    # 3) tmp_dir is a file    -> user owns the file, we write exactly there (no cleanup)

    tdir = None  # Assume caller owns the directory
    cleanup_file = False
    if not tmp_dir:
        # we own the directory. Best effort cleanup on Windows:
        # https://www.scivision.dev/python-tempfile-permission-error-windows/
        tdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        outfile = _mktemp_in(tdir.name)
    elif os.path.isdir(tmp_dir):
        outfile = _mktemp_in(tmp_dir)
        cleanup_file = True  # we own the file
    else:
        # explicit file path
        outfile = tmp_dir

    try:
        # Write arrow file
        logger.debug(f"Writing table to temporary file: {outfile}")
        _write_arrow_file(outfile, pa_table, pa_schema, max_chunksize)

        # Send to Arize
        logger.debug(
            "Uploading file to Arize",
            extra={"path": outfile, "size_bytes": _filesize(outfile)},
        )
        # Post file
        with open(outfile, "rb") as f:
            resp = requests.post(
                files_url,
                timeout=timeout,
                data=f,
                headers=headers,
                verify=verify,
            )
            if resp.status_code in (401, 403):
                raise AuthenticationError(
                    status_code=resp.status_code,
                    message=resp.text,
                )
            if not (200 <= resp.status_code < 300):
                raise APIError(
                    status_code=resp.status_code,
                    message=resp.text,
                )
            _maybe_log_project_url(resp)
            return resp
    finally:
        if tdir is not None:
            try:
                # triggers TemporaryDirectory cleanup (best-effort on Windows)
                tdir.cleanup()  # cleaning the entire dir, no need to clean the file
            except Exception as e:
                logger.warning(
                    f"Failed to remove temporary directory {tdir.name}: {e!s}"
                )
        elif cleanup_file:
            try:
                os.remove(outfile)
            except Exception as e:
                logger.warning(
                    f"Failed to remove temporary file {outfile}: {e!s}"
                )


def split_batches_by_byte_budget(
    batches: Iterable[pa.RecordBatch],
    max_batch_bytes: int = DEFAULT_FLIGHT_BATCH_BUDGET_BYTES,
    max_batch_rows: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Re-cut a stream of record batches so each fits the byte and row limits.

    Works one input batch at a time and never holds more than one of them, so
    a file-backed stream stays file-backed. Within each input batch, rows
    accumulate into an output batch until their running size would cross
    ``max_batch_bytes``, at which point the batch is cut. Sizes come from
    PyArrow's own accounting on zero-copy slices, so rows whose widths vary by
    orders of magnitude still yield batches within the budget. Input batches
    are never merged, so an output batch is never larger than its input.

    Args:
        batches: Record batches in row order.
        max_batch_bytes: Byte budget for a single batch. Defaults to
            DEFAULT_FLIGHT_BATCH_BUDGET_BYTES.
        max_batch_rows: Optional ceiling on the rows in a single batch. The byte
            budget applies on top of it. Defaults to :obj:`None` (no ceiling).

    Yields:
        pa.RecordBatch: Batches in row order, together covering every row.

    A single row larger than the Flight server limit is yielded on its own for
    the server to reject. The server commits nothing from a stream it fails,
    but commits every batch already sent when the client closes the stream.

    Raises:
        ValueError: If max_batch_bytes or max_batch_rows is below 1, or
            max_batch_bytes is above the Flight server limit.
    """
    if max_batch_bytes < 1:
        raise ValueError(
            f"max_batch_bytes must be at least 1, got {max_batch_bytes}"
        )
    if max_batch_bytes > FLIGHT_SERVER_MAX_MESSAGE_BYTES:
        raise ValueError(
            f"max_batch_bytes must be at most the Flight server limit of "
            f"{FLIGHT_SERVER_MAX_MESSAGE_BYTES} bytes, got {max_batch_bytes}"
        )
    if max_batch_rows is not None and max_batch_rows < 1:
        raise ValueError(
            f"max_batch_rows must be at least 1, got {max_batch_rows}"
        )
    return _split_batches(batches, max_batch_bytes, max_batch_rows)


def _split_batches(
    batches: Iterable[pa.RecordBatch],
    max_batch_bytes: int,
    max_batch_rows: int | None,
) -> Iterator[pa.RecordBatch]:
    for source in batches:
        step = max_batch_rows or max(source.num_rows, 1)
        for offset in range(0, source.num_rows, step):
            batch = source.slice(offset, step)
            if batch.nbytes <= max_batch_bytes:
                yield batch
                continue
            yield from _split_batch_by_bytes(batch, max_batch_bytes)


def _split_batch_by_bytes(
    batch: pa.RecordBatch, max_batch_bytes: int
) -> Iterator[pa.RecordBatch]:
    """Cut a record batch wherever one more row would cross the byte budget.

    Each candidate slice is measured as a whole, so buffers a slice shares
    across its rows, such as a dictionary, are counted once rather than per row.

    Args:
        batch: The record batch to split.
        max_batch_bytes: Byte budget for a single batch.

    Yields:
        pa.RecordBatch: Slices in row order. A row larger than the budget is
            emitted on its own.
    """
    start = 0
    for index in range(1, batch.num_rows):
        rows = index - start + 1
        if batch.slice(start, rows).nbytes > max_batch_bytes:
            yield batch.slice(start, rows - 1)
            start = index
    yield batch.slice(start, batch.num_rows - start)


def _append_to_pyarrow_metadata(
    pa_schema: pa.Schema, new_metadata: dict[str, Any]
) -> object:
    """Append metadata to a PyArrow schema without overwriting existing keys.

    Args:
        pa_schema: The PyArrow schema to add metadata to.
        new_metadata: Dictionary of metadata key-value pairs to append.

    Returns:
        pa.Schema: A new PyArrow schema with the merged metadata.

    Raises:
        KeyError: If any keys in new_metadata conflict with existing schema metadata.
    """
    # Ensure metadata is handled correctly, even if initially None.
    metadata = pa_schema.metadata
    if metadata is None:
        # Initialize an empty dict if schema metadata was None
        metadata = {}

    conflicting_keys = metadata.keys() & new_metadata.keys()
    if conflicting_keys:
        raise KeyError(
            "Cannot append metadata to pyarrow schema. "
            f"There are conflicting keys: {log_a_list(conflicting_keys, join_word='and')}"
        )

    updated_metadata = metadata.copy()
    updated_metadata.update(new_metadata)
    return pa_schema.with_metadata(updated_metadata)


def _write_arrow_file(
    path: str, pa_table: pa.Table, pa_schema: pa.Schema, max_chunksize: int
) -> None:
    """Write a PyArrow table to an Arrow IPC file with specified schema and chunk size.

    Args:
        path: The file path where the Arrow file will be written.
        pa_table: The PyArrow table containing the data to write.
        pa_schema: The PyArrow schema to use for the file.
        max_chunksize: Maximum number of rows per record batch chunk.
    """
    with (
        pa.OSFile(path, mode="wb") as sink,
        pa.ipc.RecordBatchStreamWriter(sink, pa_schema) as writer,
    ):
        writer.write_table(pa_table, max_chunksize)


def _maybe_log_project_url(response: requests.Response) -> None:
    """Attempt to extract and log the Arize project URL from an HTTP response.

    Args:
        response: The HTTP response object from an Arize API request.

    Notes:
        Logs success message with URL if extraction succeeds, or warning if it fails.
        This function never raises exceptions.
    """
    try:
        url = get_arize_project_url(response)
        if url:
            logger.info("✅ Success! Check out your data at %s", url)
        else:
            logger.debug(
                "Upload completed without a project URL in the response. "
                "Verify ingestion in the Arize UI."
            )
    except Exception as e:
        logger.warning("Failed to get project URL: %s", e)


def _mktemp_in(directory: str) -> str:
    """Create a unique temp file path inside `directory` without leaving an open file descriptor.

    Windows-safe. The file exists on disk and is closed; caller can open/write it later.
    """
    with tempfile.NamedTemporaryFile(
        dir=directory,
        prefix="arize-",
        suffix=".arrow",
        delete=False,  # important on Windows: don't keep the file open
    ) as f:
        return f.name  # file is closed when we exit the context


def _filesize(path: str) -> int:
    """Get the size of a file in bytes.

    Args:
        path: The file path to check.

    Returns:
        int: The file size in bytes, or -1 if the file cannot be accessed.
    """
    try:
        return os.path.getsize(path)
    except Exception:
        return -1
