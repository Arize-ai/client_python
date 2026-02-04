"""Pre-release feature management and gating for the Arize SDK."""

import functools
import logging
from collections.abc import Callable
from enum import Enum
from typing import TypeVar, cast

from arize.version import __version__

logger = logging.getLogger(__name__)


class ReleaseStage(Enum):
    """Enum representing the release stage of API features."""

    ALPHA = "alpha"
    BETA = "beta"


_WARNED: set[str] = set()

_F = TypeVar("_F", bound=Callable)


def _format_prerelease_message(*, key: str, stage: ReleaseStage) -> str:
    article = "an" if stage is ReleaseStage.ALPHA else "a"
    return (
        f"[{stage.value.upper()}] {key} is {article} {stage.value} API "
        f"in Arize SDK v{__version__} and may change without notice."
    )


def prerelease_endpoint(*, key: str, stage: ReleaseStage) -> Callable[[_F], _F]:
    """Decorate a method to emit a prerelease warning via logging once per process."""

    def deco(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            if key not in _WARNED:
                _WARNED.add(key)
                logger.warning(_format_prerelease_message(key=key, stage=stage))
            return fn(*args, **kwargs)

        # Cast: functools.wraps preserves function signature at runtime but mypy can't verify this
        return cast("_F", wrapper)

    return deco
