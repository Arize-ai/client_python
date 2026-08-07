"""Typed sentinel for omitted PATCH arguments."""

from typing import Final, TypeGuard, TypeVar, final

T = TypeVar("T")


@final
class UNSET:
    """Distinguish an omitted argument from an explicit ``None`` value."""

    __slots__ = ()


_UNSET: Final[UNSET] = UNSET()


def is_provided(value: T | UNSET) -> TypeGuard[T]:
    """Return whether a PATCH argument was explicitly supplied."""
    return not isinstance(value, UNSET)
