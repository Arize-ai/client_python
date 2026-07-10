"""Unit tests for experiment run id generation in experiments/types.py."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import RFC_4122, UUID

from arize.experiments.types import _exp_id

if TYPE_CHECKING:
    import pytest


def test_exp_id_is_valid_uuid7() -> None:
    parsed = UUID(_exp_id())

    assert parsed.version == 7
    assert parsed.variant == RFC_4122


def test_exp_id_embeds_current_millisecond_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_ms = 1_700_000_000_123
    monkeypatch.setattr(time, "time", lambda: frozen_ms / 1000)

    parsed = UUID(_exp_id())

    # The first 48 bits of a UUIDv7 are the big-endian unix millisecond timestamp.
    embedded_ms = int.from_bytes(parsed.bytes[:6], "big")
    assert embedded_ms == frozen_ms


def test_exp_id_is_time_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    def make_at(ms: int) -> str:
        monkeypatch.setattr(time, "time", lambda: ms / 1000)
        return _exp_id()

    earlier = make_at(1_700_000_000_000)
    later = make_at(1_700_000_001_000)

    # UUIDv7 is time-ordered: later ids sort lexicographically after earlier ones,
    # which is what our keyset pagination relies on.
    assert earlier < later


def test_exp_id_is_unique_across_many_calls() -> None:
    # The previous EXP_ID_<6hex> scheme drew only 24 random bits and reached
    # ~50% birthday-collision odds at ~4,800 ids in a single experiment. UUIDv7
    # must not collide at a scale that would have exposed the old bug.
    n = 100_000
    ids = {_exp_id() for _ in range(n)}
    assert len(ids) == n
