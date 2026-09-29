"""Apache Arrow serialization exceptions."""

from __future__ import annotations

_BYTES_PER_MB = 1024 * 1024


class RecordBatchTooLargeError(Exception):
    """Raised when an Arrow record batch cannot fit in one Flight message."""

    def __init__(
        self, num_rows: int, nbytes: int, limit_bytes: int, budget_bytes: int
    ) -> None:
        """Initialize the exception with the offending batch's measurements.

        Args:
            num_rows: Number of rows in the batch.
            nbytes: Measured Arrow size of the batch, in bytes.
            limit_bytes: Largest message the Arize Flight server accepts.
            budget_bytes: Byte budget the batch was built against.
        """
        self.num_rows = num_rows
        self.nbytes = nbytes
        self.limit_bytes = limit_bytes
        self.budget_bytes = budget_bytes
        plural = "" if num_rows == 1 else "s"
        remedy = (
            "A single row cannot be split any further: shorten the widest "
            "column values for that row before uploading."
            if num_rows == 1
            else "Lower the byte budget, max_batch_bytes, currently "
            f"{budget_bytes} bytes, so batches stay under the limit."
        )
        super().__init__(
            f"An Arrow record batch of {num_rows} row{plural} measures "
            f"{nbytes} bytes ({nbytes / _BYTES_PER_MB:.2f} MB), over the "
            f"Arize Flight server limit of {limit_bytes} bytes "
            f"({limit_bytes / _BYTES_PER_MB:.2f} MB). {remedy}"
        )

    def __reduce__(
        self,
    ) -> tuple[type[RecordBatchTooLargeError], tuple[int, int, int, int]]:
        """Return the arguments needed to rebuild this exception when unpickled."""
        return (
            self.__class__,
            (self.num_rows, self.nbytes, self.limit_bytes, self.budget_bytes),
        )
