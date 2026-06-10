import sys

from arize.exceptions.config import (
    InvalidDefaultHeadersError,
)
from arize.version import __version__

SDK_LANGUAGE = "python"
SDK_PACKAGE_NAME = "arize"
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _builtin_sdk_metadata() -> dict[str, str]:
    """Return the SDK identity metadata shared across all transports."""
    return {
        "sdk-language": SDK_LANGUAGE,
        "language-version": PYTHON_VERSION,
        "sdk-version": __version__,
        "sdk-package-name": SDK_PACKAGE_NAME,
    }


def _builtin_http_headers(api_key: str) -> dict[str, str]:
    """Return the SDK's built-in HTTP REST headers."""
    return {
        "authorization": api_key,
        **_builtin_sdk_metadata(),
    }


def _builtin_grpc_headers(api_key: str) -> dict[str, str]:
    """Return the SDK's built-in grpc-gateway headers (``Grpc-Metadata-`` prefixed)."""
    prefixed = {
        f"Grpc-Metadata-{key}": value
        for key, value in _builtin_sdk_metadata().items()
    }
    return {
        "authorization": api_key,
        **prefixed,
    }


def _builtin_flight_headers(api_key: str) -> dict[str, str]:
    """Return the SDK's built-in Apache Arrow Flight headers.

    ``auth-token-bin`` is a gRPC binary ("-bin") metadata header carrying the API
    key; the Flight client byte-encodes every value before sending.
    """
    return {
        "origin": "arize-logging-client",
        "auth-token-bin": api_key,
        **_builtin_sdk_metadata(),
    }


# Casefolded union of every built-in header key across all transports. A user's
# ``default_headers`` may not collide (case-insensitively) with any of these.
#
# NOTE: checking the raw user key is sufficient only while every
# ``Grpc-Metadata-*`` built-in has a plain twin in this union (so a user key can
# never *become* a built-in collision only after gRPC prefixing). A future
# gRPC-only built-in would require re-checking the prefixed form.
_RESERVED_HEADER_KEYS_CASEFOLDED: frozenset[str] = frozenset(
    key.casefold()
    for builder in (
        _builtin_http_headers,
        _builtin_grpc_headers,
        _builtin_flight_headers,
    )
    for key in builder("")
)

_HEADER_INJECTION_CHARS = ("\r", "\n", "\0")


def _validate_default_headers(headers: dict[str, str]) -> None:
    """Validate user-supplied default headers, raising on any violation.

    Args:
        headers: The user-supplied ``default_headers`` mapping.

    Raises:
        InvalidDefaultHeadersError: If any key/value is not a string, the key is
            empty or contains whitespace, the key or value contains a control
            character, the key contains ':' or starts with 'Grpc-Metadata-', the
            key is not ASCII or the value is not Latin-1 encodable (HTTP wire
            constraints), or the key collides (case-insensitively) with a
            built-in SDK header.
    """
    for k, v in headers.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise InvalidDefaultHeadersError(
                "default_headers keys and values must be strings; got key "
                f"{k!r} ({type(k).__name__}) with value of type "
                f"{type(v).__name__}."
            )
        if not k:
            raise InvalidDefaultHeadersError(
                "default_headers keys must be non-empty."
            )
        for label, text in (("key", k), ("value", v)):
            if any(c in text for c in _HEADER_INJECTION_CHARS):
                raise InvalidDefaultHeadersError(
                    f"default_headers {label} {text!r} contains an illegal "
                    "control character (\\r, \\n, or \\0)."
                )
        # These headers flow over all three transports, so we validate against
        # the most restrictive one. Flight byte-encodes values as UTF-8 and would
        # accept arbitrary Unicode, but the HTTP transports (REST via urllib3,
        # grpc-gateway via requests) both bottom out in Python's ``http.client``,
        # which encodes header names as ASCII and values as Latin-1 and raises
        # UnicodeEncodeError at send time otherwise. Latin-1 is therefore the
        # lowest common denominator: rejecting here turns a late, transport-
        # dependent crash into one clear error at construction time.
        if not k.isascii():
            raise InvalidDefaultHeadersError(
                f"default_headers key {k!r} must contain only ASCII characters."
            )
        try:
            v.encode("latin-1")
        except UnicodeEncodeError as e:
            raise InvalidDefaultHeadersError(
                f"default_headers value {v!r} for key {k!r} must be Latin-1 "
                "encodable; HTTP header values cannot carry arbitrary Unicode."
            ) from e
        if any(c.isspace() for c in k):
            raise InvalidDefaultHeadersError(
                f"default_headers key {k!r} must not contain whitespace; "
                "http.client rejects header names with spaces at send time."
            )
        if ":" in k:
            raise InvalidDefaultHeadersError(
                f"default_headers key {k!r} must not contain ':'."
            )
        if k.casefold().startswith("grpc-metadata-"):
            raise InvalidDefaultHeadersError(
                f"default_headers key {k!r} must not start with 'Grpc-Metadata-'; "
                "the SDK adds this prefix automatically for the gRPC transport."
            )
        if k.casefold() in _RESERVED_HEADER_KEYS_CASEFOLDED:
            raise InvalidDefaultHeadersError(
                f"default_headers key {k!r} is reserved by the SDK and cannot "
                "be overridden."
            )
