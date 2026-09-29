"""PyArrow-related constants for data processing."""

MAX_CHUNKSIZE = 100_000

# The Arize Flight server rejects any inbound gRPC message above this size.
FLIGHT_SERVER_MAX_MESSAGE_BYTES = 512 * 1024 * 1024
# Leaves room under the server limit for Arrow IPC framing and proto overhead.
DEFAULT_FLIGHT_BATCH_BUDGET_BYTES = 256 * 1024 * 1024
