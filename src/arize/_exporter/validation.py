from __future__ import annotations

from typing import TYPE_CHECKING

from arize.ml.types import Environments

if TYPE_CHECKING:
    from datetime import datetime


def validate_input_type(
    input: object,
    input_name: str,
    input_type: type,
    allow_none: bool = False,
) -> None:
    if input is None:
        if allow_none:
            return
        raise TypeError(
            f"{input_name} {input} is type {type(input)}, but must not be None"
        )

    if isinstance(input, input_type):
        return

    raise TypeError(
        f"{input_name} {input} is type {type(input)}, but must be a {input_type.__name__}"
    )


def validate_input_value(
    input: object,
    input_name: str,
    choices: tuple,
) -> None:
    if input in choices:
        return
    raise ValueError(
        f"{input_name} is {input}, but must be one of {', '.join(str(c) for c in choices)}"
    )


def validate_start_end_time(start_time: datetime, end_time: datetime) -> None:
    if start_time >= end_time:
        raise ValueError("start_time must be before end_time")


# Must match druid.MinHashSampleRate.
MIN_SAMPLE_RATE = 1e-6


def validate_sample_rate(
    sample_rate: float | None, environment: Environments
) -> None:
    if sample_rate is None:
        return
    if isinstance(sample_rate, bool) or not isinstance(
        sample_rate, (int, float)
    ):
        raise TypeError(
            f"sample_rate {sample_rate} is type {type(sample_rate)}, "
            "but must be a float"
        )
    if not MIN_SAMPLE_RATE <= sample_rate <= 1:
        raise ValueError(
            f"sample_rate is {sample_rate}, but must be in the range "
            f"[{MIN_SAMPLE_RATE:.2g}, 1]"
        )
    if sample_rate < 1 and environment != Environments.TRACING:
        raise ValueError(
            "sample_rate is only supported for the Tracing environment"
        )
