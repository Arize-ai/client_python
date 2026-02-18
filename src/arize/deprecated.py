"""Deprecation management for the Arize SDK."""

import functools
import logging
import warnings
from collections.abc import Callable
from typing import TypeVar, cast

from arize.version import __version__

logger = logging.getLogger(__name__)


_WARNED: set[str] = set()

_F = TypeVar("_F", bound=Callable)


def _format_deprecation_message(
    *,
    key: str,
    reason: str = "",
    alternative: str = "",
) -> str:
    """Format a deprecation warning message.

    Args:
        key: The identifier for the deprecated feature (e.g., "datasets.old_method").
        reason: Optional explanation of why it's deprecated.
        alternative: Optional suggestion for what to use instead.

    Returns:
        A formatted deprecation message string.
    """
    msg = f"[DEPRECATED] {key} is deprecated in Arize SDK v{__version__}"

    if reason:
        msg += f" ({reason})"

    if alternative:
        msg += f". Use {alternative} instead"

    msg += "."

    return msg


def deprecated(
    *,
    key: str,
    reason: str = "",
    alternative: str = "",
    emit_warning: bool = True,
) -> Callable[[_F], _F]:
    """Decorate a method to emit a deprecation warning once per process.

    This decorator logs a deprecation message and optionally emits a Python
    DeprecationWarning the first time the decorated method is called.

    Args:
        key: Unique identifier for the deprecated feature (e.g., "datasets.old_method").
        reason: Optional explanation of why the feature is deprecated.
        alternative: Optional suggestion for what to use instead.
        emit_warning: If True (default), emit a DeprecationWarning in addition to logging.

    Returns:
        A decorator function that wraps the target method.
    """

    def deco(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            if key not in _WARNED:
                _WARNED.add(key)
                msg = _format_deprecation_message(
                    key=key, reason=reason, alternative=alternative
                )
                logger.warning(msg)

                if emit_warning:
                    warnings.warn(msg, DeprecationWarning, stacklevel=2)

            return fn(*args, **kwargs)

        # Cast: functools.wraps preserves function signature at runtime but mypy can't verify this
        return cast("_F", wrapper)

    return deco
