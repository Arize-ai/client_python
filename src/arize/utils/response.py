"""Checks for response values needed by SDK operations."""

from typing import TypeVar

from pydantic import BaseModel

from arize.exceptions import MissingResponseFieldError

T = TypeVar("T")


def require_operation_value(value: T | None, *, field: str) -> T:
    """Return a required operation value or explain why it is unavailable."""
    if value is None:
        raise MissingResponseFieldError(field)
    return value


def coerce_response_role(
    value: object,
    *,
    wrapper_type: type[BaseModel],
    predefined_response_type: type[BaseModel],
    custom_response_type: type[BaseModel],
    predefined_type: type[BaseModel],
    custom_type: type[BaseModel],
) -> object:
    """Convert a response role to the domain type used by SDK requests."""
    if not isinstance(value, wrapper_type):
        return value
    actual = getattr(value, "actual_instance", None)
    # Domain roles are request inputs: predefined roles use a name, custom roles use an ID.
    if isinstance(actual, predefined_response_type):
        return predefined_type.model_validate(
            {
                "name": require_operation_value(
                    getattr(actual, "name", None), field="role.name"
                )
            }
        )
    if isinstance(actual, custom_response_type):
        return custom_type.model_validate(
            {
                "id": require_operation_value(
                    getattr(actual, "id", None), field="role.id"
                ),
                "name": getattr(actual, "name", None),
            }
        )
    if actual is None:
        raise MissingResponseFieldError("role.actual_instance")
    raise TypeError(f"Unknown role type: {type(actual)!r}")
