from __future__ import annotations

from typing import TYPE_CHECKING

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
