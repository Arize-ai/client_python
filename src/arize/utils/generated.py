"""Utilities for working with OpenAPI-generated discriminated-union types."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from pydantic import BaseModel

_T = TypeVar("_T", bound="BaseModel")


def unwrap_discriminated_union(
    wrapper: object,
    type_map: dict[type[BaseModel], type[_T]],
) -> _T:
    """Unwrap a generated oneOf wrapper into a domain subclass.

    The generated OpenAPI client wraps ``oneOf`` types in a container that
    stores the resolved instance in an ``actual_instance`` attribute.  This
    helper inspects that instance, finds its matching entry in ``type_map``,
    and returns a new instance of the mapped domain subclass.

    Args:
        wrapper: A generated oneOf wrapper object with an ``actual_instance``
            attribute (e.g. ``UserRoleAssignment``).
        type_map: Mapping from generated base class → domain subclass to
            construct.  Evaluated in insertion order; the first matching key
            wins.

    Returns:
        An instance of the matching domain subclass.

    Raises:
        TypeError: If ``actual_instance`` does not match any key in
            ``type_map``.
    """
    actual = getattr(wrapper, "actual_instance", None)
    for generated_cls, domain_cls in type_map.items():
        if isinstance(actual, generated_cls):
            return domain_cls.model_validate(actual.model_dump())  # type: ignore[attr-defined]
    raise TypeError(
        f"Unknown type in discriminated union: {type(actual)!r}. "
        f"Expected one of: {[c.__name__ for c in type_map]}"
    )
