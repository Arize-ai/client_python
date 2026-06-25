import os

ALLOWED_HTTP_SCHEMES = {"http", "https"}


def _env_http_scheme(name: str, default: str) -> str:
    """Get an HTTP scheme from environment variable with validation.

    Args:
        name: The environment variable name.
        default: The default value if the environment variable is not set.

    Returns:
        str: The validated HTTP scheme ('http' or 'https').

    Raises:
        ValueError: If the scheme is not 'http' or 'https'.
    """
    v = _env_str(name, default).lower()
    if v not in ALLOWED_HTTP_SCHEMES:
        raise ValueError(
            f"{name} must be one of {sorted(ALLOWED_HTTP_SCHEMES)}. Found {v!r}"
        )
    return v


def _env_str(
    name: str,
    default: str,
    min_len: int | None = None,
    max_len: int | None = None,
) -> str:
    """Get a string value from environment variable with length validation.

    Args:
        name: The environment variable name.
        default: The default value if the environment variable is not set.
        min_len: Optional minimum length constraint for the string.
        max_len: Optional maximum length constraint for the string.

    Returns:
        str: The validated string value (stripped of whitespace).

    Raises:
        ValueError: If the string length violates min_len or max_len constraints.
    """
    val = os.getenv(name, default).strip()

    if min_len is not None and len(val) < min_len:
        raise ValueError(
            f"The value of environment variable {name} must be at least {min_len} "
            f"characters long. Found {len(val)} characters."
        )
    if max_len is not None and len(val) > max_len:
        raise ValueError(
            f"The value of environment variable {name} must be at most {max_len} "
            f"characters long. Found {len(val)} characters."
        )
    return val


def _env_int(
    name: str,
    default: int,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    """Get an integer value from environment variable with range validation.

    Args:
        name: The environment variable name.
        default: The default value if the environment variable is not set.
        min_val: Optional minimum value constraint for the integer.
        max_val: Optional maximum value constraint for the integer.

    Returns:
        int: The validated integer value.

    Raises:
        ValueError: If the value cannot be parsed as an integer or violates min_val/max_val constraints.
    """
    raw = os.getenv(name, default)
    try:
        val = int(raw)
    except Exception as e:
        raise ValueError(
            f"Environment variable {name} must be an int. Found: {raw!r}"
        ) from e

    if min_val is not None and val < min_val:
        raise ValueError(
            f"The value of environment variable {name} must be at least {min_val}. Found {val}."
        )
    if max_val is not None and val > max_val:
        raise ValueError(
            f"The value of environment variable {name} must be at most {max_val}. Found {val}."
        )
    return val


def _env_float(
    name: str,
    default: float,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Get a float value from environment variable with range validation.

    Args:
        name: The environment variable name.
        default: The default value if the environment variable is not set.
        min_val: Optional minimum value constraint for the float.
        max_val: Optional maximum value constraint for the float.

    Returns:
        float: The validated float value.

    Raises:
        ValueError: If the value cannot be parsed as a float or violates min_val/max_val constraints.
    """
    raw = os.getenv(name, default)
    try:
        val = float(raw)
    except Exception as e:
        raise ValueError(
            f"Environment variable {name} must be a float. Found: {raw!r}"
        ) from e

    if min_val is not None and val < min_val:
        raise ValueError(
            f"The value of environment variable {name} must be at least {min_val}. Found {val}."
        )
    if max_val is not None and val > max_val:
        raise ValueError(
            f"The value of environment variable {name} must be at most {max_val}. Found {val}."
        )
    return val


def _env_bool(name: str, default: bool) -> bool:
    """Get a boolean value from environment variable.

    Args:
        name: The environment variable name.
        default: The default boolean value if the environment variable is not set.

    Returns:
        bool: The parsed boolean value.
    """
    return _parse_bool(os.getenv(name, str(default)))


def _env_str_fallback(*names: str, default: str = "") -> str:
    """Get a string value from the first set environment variable in a priority list.

    Checks each name in order and returns the first non-empty value found.
    Useful for honouring familiar env vars alongside Arize-specific overrides
    (e.g. ARIZE_SSL_CA_CERT → REQUESTS_CA_BUNDLE → SSL_CERT_FILE).

    Args:
        *names: Environment variable names to check, in priority order.
        default: The default value if none of the variables are set.

    Returns:
        str: The first non-empty value found among the env vars, or default.
    """
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return default


def _parse_bool(val: bool | str | None) -> bool:
    """Parse a boolean value from various input types.

    Args:
        val: The value to parse. Can be a bool, string, or None.

    Returns:
        bool: True if the value is already True or matches one of the truthy strings
            ('1', 'true', 'yes', 'on', case-insensitive). False otherwise.
    """
    if isinstance(val, bool):
        return val
    return (val or "").strip().lower() in {"1", "true", "yes", "on"}
