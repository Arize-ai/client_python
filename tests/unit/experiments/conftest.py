"""Experiments test fixtures — patches heavy transitive imports before collection."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ExperimentsClient imports ArizeFlightClient at module level, which pulls in
# protobuf-generated code that may not be available in lightweight test envs.
# Stub the flight module out before pytest collects (and therefore imports)
# the test file so that `from arize.experiments.client import ExperimentsClient` works.
for _mod in ("arize._flight", "arize._flight.client", "arize._flight.types"):
    sys.modules.setdefault(_mod, MagicMock())
