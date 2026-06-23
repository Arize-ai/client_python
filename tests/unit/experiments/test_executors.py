"""Unit tests for experiment evaluator executors."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from arize.experiments.evaluators import executors

if TYPE_CHECKING:
    import pytest


def test_running_event_loop_warning_omits_nest_asyncio_on_python_3_14(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running-loop fallback should not recommend patching asyncio on 3.14+."""
    monkeypatch.setattr(executors.sys, "version_info", (3, 14, 0))

    def sync_fn(payload: object) -> object:
        return payload

    async def async_fn(payload: object) -> object:
        return payload

    caplog.set_level(
        logging.WARNING,
        logger="arize.experiments.evaluators.executors",
    )

    async def get_executor_inside_running_loop() -> object:
        return executors.get_executor_on_sync_context(sync_fn, async_fn)

    executor = asyncio.run(get_executor_inside_running_loop())

    assert isinstance(executor, executors.SyncExecutor)
    assert "existing event loop" in caplog.text
    assert "Falling back to sync execution" in caplog.text
    assert "nest_asyncio" not in caplog.text


def test_running_event_loop_warning_recommends_nest_asyncio_before_python_3_14(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running-loop fallback can still mention nest_asyncio before 3.14."""
    monkeypatch.setattr(executors.sys, "version_info", (3, 13, 0))

    def sync_fn(payload: object) -> object:
        return payload

    async def async_fn(payload: object) -> object:
        return payload

    caplog.set_level(
        logging.WARNING,
        logger="arize.experiments.evaluators.executors",
    )

    async def get_executor_inside_running_loop() -> object:
        return executors.get_executor_on_sync_context(sync_fn, async_fn)

    executor = asyncio.run(get_executor_inside_running_loop())

    assert isinstance(executor, executors.SyncExecutor)
    assert "nest_asyncio.apply()" in caplog.text
