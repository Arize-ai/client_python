"""SDK configuration and settings management for the Arize client."""

import logging
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

from arize.constants.config import (
    DEFAULT_API_HOST,
    DEFAULT_API_SCHEME,
    DEFAULT_ARIZE_DIRECTORY,
    DEFAULT_ENABLE_CACHING,
    DEFAULT_FLIGHT_HOST,
    DEFAULT_FLIGHT_PORT,
    DEFAULT_FLIGHT_SCHEME,
    DEFAULT_MAX_HTTP_PAYLOAD_SIZE_MB,
    DEFAULT_OTLP_HOST,
    DEFAULT_OTLP_SCHEME,
    DEFAULT_PYARROW_MAX_CHUNKSIZE,
    DEFAULT_REQUEST_VERIFY,
    DEFAULT_STREAM_MAX_QUEUE_BOUND,
    DEFAULT_STREAM_MAX_WORKERS,
    ENV_API_HOST,
    ENV_API_KEY,
    ENV_API_SCHEME,
    ENV_ARIZE_DIRECTORY,
    ENV_BASE_DOMAIN,
    ENV_ENABLE_CACHING,
    ENV_FLIGHT_HOST,
    ENV_FLIGHT_PORT,
    ENV_FLIGHT_SCHEME,
    ENV_MAX_HTTP_PAYLOAD_SIZE_MB,
    ENV_OTLP_HOST,
    ENV_OTLP_SCHEME,
    ENV_PYARROW_MAX_CHUNKSIZE,
    ENV_REGION,
    ENV_REQUEST_VERIFY,
    ENV_SINGLE_HOST,
    ENV_SINGLE_PORT,
    ENV_STREAM_MAX_QUEUE_BOUND,
    ENV_STREAM_MAX_WORKERS,
)
from arize.constants.pyarrow import MAX_CHUNKSIZE
from arize.exceptions.auth import MissingAPIKeyError
from arize.exceptions.config import MultipleEndpointOverridesError
from arize.regions import REGION_ENDPOINTS, Region
from arize.version import __version__

logger = logging.getLogger(__name__)

PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
SENSITIVE_FIELD_MARKERS = ("key", "token", "secret")
ALLOWED_HTTP_SCHEMES = {"http", "https"}


def _is_sensitive_field(name: str) -> bool:
    """Check if a field name contains sensitive information markers.

    Args:
        name: The field name to check.

    Returns:
        bool: True if the field name contains 'key', 'token', or 'secret' (case-insensitive).
    """
    n = name.lower()
    return bool(any(k in n for k in SENSITIVE_FIELD_MARKERS))


def _mask_secret(secret: str, N: int = 4) -> str:
    """Mask a secret string by showing only the first N characters.

    Args:
        secret: The secret string to mask.
        N: Number of characters to show before masking. Defaults to 4.

    Returns:
        str: The masked string (first N chars + '***'), or empty string if input is empty.
    """
    if len(secret) == 0:
        return ""
    return f"{secret[:N]}***"


def _endpoint(scheme: str, base: str, path: str = "") -> str:
    """Construct a full endpoint URL from scheme, base, and optional path.

    Args:
        scheme: The URL scheme (e.g., "http", "https").
        base: The base URL or hostname.
        path: Optional path to append to the base URL. Defaults to empty string.

    Returns:
        str: The fully constructed endpoint URL.
    """
    endpoint = scheme + "://" + base.rstrip("/")
    if path:
        endpoint += "/" + path.lstrip("/")
    return endpoint


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


@dataclass(frozen=True)
class SDKConfiguration:
    """Configuration for the Arize SDK with endpoint and authentication settings.

    This class holds pure configuration data and does not manage client lifecycle.
    Client creation and caching is handled by :class:`arize.ArizeClient`.

    This class is used internally by ArizeClient to manage SDK configuration. It is not
    recommended to use this class directly; users should interact with ArizeClient
    instead.

    Each configuration parameter follows this resolution order:
        1. Explicit value passed to ArizeClient constructor (highest priority)
        2. Environment variable value
        3. Built-in default value (lowest priority)

    Args:
        api_key: Arize API key for authentication. Required.
            Environment variable: ARIZE_API_KEY.
            Default: None (must be provided via argument or environment variable).
        api_host: API endpoint host.
            Environment variable: ARIZE_API_HOST.
            Default: "api.arize.com".
        api_scheme: API endpoint scheme (http/https).
            Environment variable: ARIZE_API_SCHEME.
            Default: "https".
        otlp_host: OTLP (OpenTelemetry Protocol) endpoint host.
            Environment variable: ARIZE_OTLP_HOST.
            Default: "otlp.arize.com".
        otlp_scheme: OTLP endpoint scheme (http/https).
            Environment variable: ARIZE_OTLP_SCHEME.
            Default: "https".
        flight_host: Apache Arrow Flight endpoint host.
            Environment variable: ARIZE_FLIGHT_HOST.
            Default: "flight.arize.com".
        flight_port: Apache Arrow Flight endpoint port (1-65535).
            Environment variable: ARIZE_FLIGHT_PORT.
            Default: 443.
        flight_scheme: Apache Arrow Flight endpoint scheme.
            Environment variable: ARIZE_FLIGHT_SCHEME.
            Default: "grpc+tls".
        pyarrow_max_chunksize: Maximum chunk size for PyArrow operations (1 to MAX_CHUNKSIZE).
            Environment variable: ARIZE_MAX_CHUNKSIZE.
            Default: 10_000.
        request_verify: Whether to verify SSL certificates for HTTP requests.
            Environment variable: ARIZE_REQUEST_VERIFY.
            Default: True.
        stream_max_workers: Maximum number of worker threads for streaming operations (minimum: 1).
            Environment variable: ARIZE_STREAM_MAX_WORKERS.
            Default: 8.
        stream_max_queue_bound: Maximum queue size for streaming operations (minimum: 1).
            Environment variable: ARIZE_STREAM_MAX_QUEUE_BOUND.
            Default: 5000.
        max_http_payload_size_mb: Maximum HTTP payload size in megabytes (minimum: 1).
            Environment variable: ARIZE_MAX_HTTP_PAYLOAD_SIZE_MB.
            Default: 100.
        arize_directory: Directory for Arize SDK files (cache, logs, etc.).
            Environment variable: ARIZE_DIRECTORY.
            Default: "~/.arize".
        enable_caching: Whether to enable local caching.
            Environment variable: ARIZE_ENABLE_CACHING.
            Default: True.
        region: Arize region (e.g., US_CENTRAL, EU_WEST). When specified, overrides
            individual host/port settings.
            Environment variable: ARIZE_REGION.
            Default: :class:`Region.UNSET`.
        single_host: Single host to use for all endpoints. When specified, overrides
            individual host settings.
            Environment variable: ARIZE_SINGLE_HOST.
            Default: "" (not set).
        single_port: Single port to use for all endpoints. When specified, overrides
            individual port settings (0-65535).
            Environment variable: ARIZE_SINGLE_PORT.
            Default: 0 (not set).
        base_domain: Base domain for generating all endpoint hosts. Intended for Private Connect
            setups. When specified, generates hosts as api.<base_domain>, otlp.<base_domain>,
            flight.<base_domain>. When specified, overrides individual host settings.
            Environment variable: ARIZE_BASE_DOMAIN.
            Default: "" (not set).

    Note:
        The endpoint override options (region, single_host/single_port, base_domain) are
        mutually exclusive. Specifying more than one will raise MultipleEndpointOverridesError.

    Raises:
        MissingAPIKeyError: If api_key is not provided via argument or environment variable.
        MultipleEndpointOverridesError: If multiple endpoint override options are provided.
    """

    api_key: str = field(
        default_factory=lambda: _env_str(ENV_API_KEY, ""),
    )
    api_host: str = field(
        default_factory=lambda: _env_str(ENV_API_HOST, DEFAULT_API_HOST)
    )
    api_scheme: str = field(
        default_factory=lambda: _env_http_scheme(
            ENV_API_SCHEME,
            DEFAULT_API_SCHEME,
        ),
    )
    otlp_host: str = field(
        default_factory=lambda: _env_str(ENV_OTLP_HOST, DEFAULT_OTLP_HOST)
    )
    otlp_scheme: str = field(
        default_factory=lambda: _env_http_scheme(
            ENV_OTLP_SCHEME,
            DEFAULT_OTLP_SCHEME,
        ),
    )
    flight_host: str = field(
        default_factory=lambda: _env_str(ENV_FLIGHT_HOST, DEFAULT_FLIGHT_HOST)
    )
    flight_port: int = field(
        default_factory=lambda: _env_int(
            ENV_FLIGHT_PORT, DEFAULT_FLIGHT_PORT, min_val=1, max_val=65535
        )
    )
    flight_scheme: str = field(
        default_factory=lambda: _env_str(
            ENV_FLIGHT_SCHEME,
            DEFAULT_FLIGHT_SCHEME,
        ),
    )
    pyarrow_max_chunksize: int = field(
        default_factory=lambda: _env_int(
            ENV_PYARROW_MAX_CHUNKSIZE,
            DEFAULT_PYARROW_MAX_CHUNKSIZE,
            min_val=1,
            max_val=MAX_CHUNKSIZE,
        )
    )
    request_verify: bool = field(
        default_factory=lambda: _env_bool(
            ENV_REQUEST_VERIFY, DEFAULT_REQUEST_VERIFY
        )
    )
    stream_max_workers: int = field(
        default_factory=lambda: _env_int(
            ENV_STREAM_MAX_WORKERS, DEFAULT_STREAM_MAX_WORKERS, min_val=1
        )
    )
    stream_max_queue_bound: int = field(
        default_factory=lambda: _env_int(
            ENV_STREAM_MAX_QUEUE_BOUND,
            DEFAULT_STREAM_MAX_QUEUE_BOUND,
            min_val=1,
        )
    )
    max_http_payload_size_mb: float = field(
        default_factory=lambda: _env_float(
            ENV_MAX_HTTP_PAYLOAD_SIZE_MB,
            DEFAULT_MAX_HTTP_PAYLOAD_SIZE_MB,
            min_val=1,
        )
    )
    arize_directory: str = field(
        default_factory=lambda: _env_str(
            ENV_ARIZE_DIRECTORY, DEFAULT_ARIZE_DIRECTORY
        )
    )
    enable_caching: bool = field(
        default_factory=lambda: _env_bool(
            ENV_ENABLE_CACHING, DEFAULT_ENABLE_CACHING
        )
    )
    region: Region = field(
        default_factory=lambda: Region(_env_str(ENV_REGION, ""))
    )
    single_host: str = field(
        default_factory=lambda: _env_str(ENV_SINGLE_HOST, "")
    )
    single_port: int = field(
        default_factory=lambda: _env_int(
            ENV_SINGLE_PORT, 0, min_val=0, max_val=65535
        )
    )
    base_domain: str = field(
        default_factory=lambda: _env_str(ENV_BASE_DOMAIN, "")
    )

    def __post_init__(self) -> None:
        """Validate and configure SDK endpoints after initialization.

        Endpoint override options are mutually exclusive. Only one of the following
        can be specified:
        1. region - Overrides all via REGION_ENDPOINTS mapping
        2. single_host/single_port - Overrides individual hosts/ports
        3. base_domain - Generates hosts from base domain

        If none are specified, per-endpoint host/port settings are used.

        Raises:
            MissingAPIKeyError: If api_key is not provided.
            MultipleEndpointOverridesError: If multiple endpoint override options are provided.
        """
        # Validate configuration
        if not self.api_key:
            raise MissingAPIKeyError()

        # Check which override options are set
        has_base_domain = bool(self.base_domain)
        has_single_host = bool(self.single_host)
        has_single_port = self.single_port != 0
        has_region = self.region is not Region.UNSET

        # Ensure only one override method is used (mutually exclusive)
        override_count = sum(
            [has_base_domain, has_single_host or has_single_port, has_region]
        )
        if override_count > 1:
            # Determine which overrides were provided
            provided_overrides = []
            if has_region:
                provided_overrides.append(f"region={self.region.value}")
            if has_single_host or has_single_port:
                if has_single_host:
                    provided_overrides.append(
                        f"single_host={self.single_host!r}"
                    )
                if has_single_port:
                    provided_overrides.append(f"single_port={self.single_port}")
            if has_base_domain:
                provided_overrides.append(f"base_domain={self.base_domain!r}")

            error_message = (
                f"Multiple endpoint override options provided: {', '.join(provided_overrides)}. "
                "Only one of the following can be specified: 'region', "
                "'single_host'/'single_port', or 'base_domain'."
            )
            logger.error(error_message)
            raise MultipleEndpointOverridesError(error_message)

        if has_base_domain:
            logger.info(
                "Base domain %r provided; generating hosts from base domain.",
                self.base_domain,
            )
            object.__setattr__(self, "api_host", f"api.{self.base_domain}")
            object.__setattr__(self, "otlp_host", f"otlp.{self.base_domain}")
            object.__setattr__(
                self, "flight_host", f"flight.{self.base_domain}"
            )

        if has_single_host:
            logger.info(
                "Single host %r provided; overriding hosts configuration with single host.",
                self.single_host,
            )
            object.__setattr__(self, "api_host", self.single_host)
            object.__setattr__(self, "otlp_host", self.single_host)
            object.__setattr__(self, "flight_host", self.single_host)

        if has_single_port:
            logger.info(
                "Single port %s provided; overriding ports configuration with single port.",
                self.single_port,
            )
            object.__setattr__(self, "flight_port", self.single_port)

        if has_region:
            logger.info(
                "Region %s provided; overriding hosts & ports configuration with region defaults.",
                self.region.value,
            )
            endpoints = REGION_ENDPOINTS[self.region]
            object.__setattr__(self, "api_host", endpoints.api_host)
            object.__setattr__(self, "otlp_host", endpoints.otlp_host)
            object.__setattr__(self, "flight_host", endpoints.flight_host)
            object.__setattr__(self, "flight_port", endpoints.flight_port)

    @property
    def cache_dir(self) -> str:
        """Return the path to the cache directory."""
        return str(Path(self.arize_directory) / "cache")

    @property
    def api_url(self) -> str:
        """Return the base API URL."""
        return _endpoint(self.api_scheme, self.api_host)

    @property
    def otlp_url(self) -> str:
        """Return the OTLP endpoint URL."""
        return _endpoint(self.otlp_scheme, self.otlp_host, "/v1")

    @property
    def files_url(self) -> str:
        """Return the files upload endpoint URL."""
        return _endpoint(self.api_scheme, self.api_host, "/v1/pandas_arrow")

    @property
    def records_url(self) -> str:
        """Return the records logging endpoint URL."""
        return _endpoint(self.api_scheme, self.api_host, "/v1/log")

    @property
    def headers(self) -> dict[str, str]:
        """Return HTTP headers for API requests."""
        # Create base headers
        return {
            "authorization": self.api_key,
            "sdk-language": "python",
            "language-version": PYTHON_VERSION,
            "sdk-version": __version__,
            # "arize-space-id": self._space_id,
            # "arize-interface": "batch",
            # "sync": "0",  # Defaults to async logging
        }

    @property
    def headers_grpc(self) -> dict[str, str]:
        """Return headers for gRPC requests."""
        return {
            "authorization": self.api_key,
            "Grpc-Metadata-sdk-language": "python",
            "Grpc-Metadata-language-version": PYTHON_VERSION,
            "Grpc-Metadata-sdk-version": __version__,
            # "Grpc-Metadata-arize-space-id": space_id,
            # "Grpc-Metadata-arize-interface": "stream",
        }

    def __repr__(self) -> str:
        """Return a detailed string representation with masked sensitive fields."""
        # Dynamically build repr for all fields
        lines = [f"{self.__class__.__name__}("]
        for f in fields(self):
            if not f.repr:
                continue
            val = getattr(self, f.name)
            if _is_sensitive_field(f.name):
                val = _mask_secret(val, 6)
            lines.append(f"  {f.name}={val!r},")
        lines.append(")")
        return "\n".join(lines)
