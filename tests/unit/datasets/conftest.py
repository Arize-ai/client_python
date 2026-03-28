"""Datasets test fixtures — patches heavy transitive imports before collection."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# DatasetsClient imports ArizeFlightClient at module level, which pulls in
# protobuf-generated code that may not be available in lightweight test envs.
# Stub the flight module out before pytest collects (and therefore imports)
# the test file so that `from arize.datasets.client import DatasetsClient` works.
for _mod in ("arize._flight", "arize._flight.client"):
    sys.modules.setdefault(_mod, MagicMock())
