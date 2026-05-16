"""Internal utilities shared across SDK client modules."""


def unwrap_oneof(wrapper: object) -> object:
    """Extract ``actual_instance`` from a generated oneOf wrapper.

    Raises:
        RuntimeError: If ``actual_instance`` is ``None``, indicating the API
            returned a wrapper that failed to resolve to a concrete type.
    """
    instance = wrapper.actual_instance  # type: ignore[attr-defined]
    if instance is None:
        raise RuntimeError(
            f"{type(wrapper).__name__} wrapper has actual_instance=None"
        )
    return instance
