"""SDK configuration and settings management for the Arize client."""

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from urllib.parse import urlparse, urlsplit

from arize._env import (
    _env_bool,
    _env_float,
    _env_http_scheme,
    _env_int,
    _env_str,
    _env_str_fallback,
)
from arize._headers import (
    _builtin_flight_headers,
    _builtin_grpc_headers,
    _builtin_http_headers,
    _validate_default_headers,
)
from arize.constants.config import (
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_API_SCHEME,
    DEFAULT_ARIZE_DIRECTORY,
    DEFAULT_ENABLE_CACHING,
    DEFAULT_FLIGHT_HOST,
    DEFAULT_FLIGHT_PORT,
    DEFAULT_FLIGHT_SCHEME,
    DEFAULT_MAX_HTTP_PAYLOAD_SIZE_MB,
    DEFAULT_MAX_PAST_YEARS,
    DEFAULT_OTLP_HOST,
    DEFAULT_OTLP_PORT,
    DEFAULT_OTLP_SCHEME,
    DEFAULT_PYARROW_MAX_CHUNKSIZE,
    DEFAULT_REQUEST_VERIFY,
    DEFAULT_STREAM_MAX_QUEUE_BOUND,
    DEFAULT_STREAM_MAX_WORKERS,
    ENV_API_HOST,
    ENV_API_KEY,
    ENV_API_PORT,
    ENV_API_SCHEME,
    ENV_ARIZE_DIRECTORY,
    ENV_BASE_DOMAIN,
    ENV_ENABLE_CACHING,
    ENV_FLIGHT_HOST,
    ENV_FLIGHT_PORT,
    ENV_FLIGHT_SCHEME,
    ENV_MAX_HTTP_PAYLOAD_SIZE_MB,
    ENV_MAX_PAST_YEARS,
    ENV_OTLP_HOST,
    ENV_OTLP_PORT,
    ENV_OTLP_SCHEME,
    ENV_PROXY_URL,
    ENV_PYARROW_MAX_CHUNKSIZE,
    ENV_REGION,
    ENV_REQUEST_VERIFY,
    ENV_SINGLE_HOST,
    ENV_SINGLE_PORT,
    ENV_SSL_CA_CERT,
    ENV_STREAM_MAX_QUEUE_BOUND,
    ENV_STREAM_MAX_WORKERS,
)
from arize.constants.pyarrow import MAX_CHUNKSIZE
from arize.exceptions.auth import MissingAPIKeyError
from arize.exceptions.config import (
    MultipleEndpointOverridesError,
)
from arize.regions import REGION_ENDPOINTS, Region

logger = logging.getLogger(__name__)

SENSITIVE_FIELD_MARKERS = ("key", "token", "secret")


def _is_sensitive_field(name: str) -> bool:
    """Check if a field name contains sensitive information markers.

    Args:
        name: The field name to check.

    Returns:
        bool: True if the field name contains 'key', 'token', or 'secret' (case-insensitive).
    """
    n = name.lower()
    return bool(any(k in n for k in SENSITIVE_FIELD_MARKERS))


def _mask_proxy_url(url: str) -> str:
    """Redact the password from a proxy URL, leaving the rest intact."""
    if not url:
        return url
    try:
        parsed = urlparse(url)
        if parsed.password:
            masked = parsed._replace(
                netloc=f"{parsed.username}:***@{parsed.hostname}"
                + (f":{parsed.port}" if parsed.port else "")
            )
            return masked.geturl()
    except Exception:  # noqa: S110
        pass
    return url


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


def _endpoint(scheme: str, base: str, path: str = "", port: int = 0) -> str:
    """Construct a full endpoint URL from scheme, base, optional path, and port.

    Args:
        scheme: The URL scheme (e.g., "http", "https").
        base: The base URL or hostname.
        path: Optional path to append to the base URL. Defaults to empty string.
        port: Optional port number. When non-zero, appended as ``:<port>`` after
            the host. When 0 (default), the port is omitted and the scheme's
            standard port applies.

    Returns:
        str: The fully constructed endpoint URL.
    """
    host = base.rstrip("/")
    if port:
        # Strip any port already embedded in base (e.g. "api.arize.com:8080") to
        # avoid producing "scheme://host:old_port:new_port".
        parsed = urlsplit(f"//{host}")
        if parsed.port is not None:
            host = parsed.hostname or host
    endpoint = scheme + "://" + host
    if port:
        endpoint += ":" + str(port)
    if path:
        endpoint += "/" + path.lstrip("/")
    return endpoint


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
        api_port: API endpoint port (0-65535). When 0 (default), the port is omitted
            from the URL and the scheme's standard port applies.
            Environment variable: ARIZE_API_PORT.
            Default: 0 (not set).
        otlp_host: OTLP (OpenTelemetry Protocol) endpoint host.
            Environment variable: ARIZE_OTLP_HOST.
            Default: "otlp.arize.com".
        otlp_scheme: OTLP endpoint scheme (http/https).
            Environment variable: ARIZE_OTLP_SCHEME.
            Default: "https".
        otlp_port: OTLP endpoint port (0-65535). When 0 (default), the port is omitted
            from the URL and the scheme's standard port applies.
            Environment variable: ARIZE_OTLP_PORT.
            Default: 0 (not set).
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
        ssl_ca_cert: Path to a CA bundle file for verifying SSL certificates.
            Useful when connecting through a proxy or on-prem gateway that terminates
            TLS with its own certificate. Reads the first set variable from:
            ARIZE_SSL_CA_CERT → REQUESTS_CA_BUNDLE → SSL_CERT_FILE.
            Default: "" (use system CAs).
        proxy_url: HTTP(S) proxy URL for all REST API and OTLP requests (e.g.
            "http://proxy.corp:8080"). Reads the first set variable from:
            ARIZE_PROXY_URL → HTTPS_PROXY → HTTP_PROXY.
            Default: "" (direct connection).
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
            api_port, otlp_port, and flight_port (0-65535).
            Environment variable: ARIZE_SINGLE_PORT.
            Default: 0 (not set).
        base_domain: Base domain for generating all endpoint hosts. Intended for Private Connect
            setups. When specified, generates hosts as api.<base_domain>, otlp.<base_domain>,
            flight.<base_domain>. When specified, overrides individual host settings.
            Environment variable: ARIZE_BASE_DOMAIN.
            Default: "" (not set).
        max_past_years: Maximum number of years in the past allowed for prediction timestamps.
            For on-prem deployments with custom retention policies, this can be increased.
            When set to a value other than the default, a warning will be issued advising
            to contact Arize support.
            Environment variable: ARIZE_MAX_PAST_YEARS.
            Default: 5.
        default_headers: Custom headers added to every outbound request across all
            transports (HTTP REST, grpc-gateway, and Apache Arrow Flight). They appear
            in the ``headers``, ``headers_grpc``, and ``headers_flight`` properties. On
            the gRPC path each key is automatically prefixed with ``Grpc-Metadata-`` so
            grpc-gateway forwards it to the backend service. Keys may not collide
            (case-insensitively) with the SDK's built-in headers, start with
            ``Grpc-Metadata-``, or contain control characters; violations raise
            InvalidDefaultHeadersError. Unlike other fields, this has no environment
            variable and is programmatic-only (argument or default).
            Default: {} (empty).

    Note:
        The endpoint override options (region, single_host/single_port, base_domain) are
        mutually exclusive. Specifying more than one will raise MultipleEndpointOverridesError.

    Raises:
        MissingAPIKeyError: If api_key is not provided via argument or environment variable.
        MultipleEndpointOverridesError: If multiple endpoint override options are provided.
        InvalidDefaultHeadersError: If default_headers contains an invalid or reserved header.
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
    api_port: int = field(
        default_factory=lambda: _env_int(
            ENV_API_PORT, DEFAULT_API_PORT, min_val=0, max_val=65535
        )
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
    otlp_port: int = field(
        default_factory=lambda: _env_int(
            ENV_OTLP_PORT, DEFAULT_OTLP_PORT, min_val=0, max_val=65535
        )
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
    ssl_ca_cert: str = field(
        default_factory=lambda: _env_str_fallback(
            ENV_SSL_CA_CERT,
            "REQUESTS_CA_BUNDLE",
            "SSL_CERT_FILE",
        )
    )
    proxy_url: str = field(
        default_factory=lambda: _env_str_fallback(ENV_PROXY_URL)
    )
    stream_max_workers: int = field(
        default_factory=lambda: _env_int(
            ENV_STREAM_MAX_WORKERS,
            DEFAULT_STREAM_MAX_WORKERS,
            min_val=1,
            max_val=32,
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
    max_past_years: int = field(
        default_factory=lambda: _env_int(
            ENV_MAX_PAST_YEARS, DEFAULT_MAX_PAST_YEARS, min_val=1
        )
    )
    default_headers: dict[str, str] = field(default_factory=dict)

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

        # Validate and freeze user-supplied default headers. Store a defensive
        # shallow copy so a caller mutating their dict afterward cannot alter
        # this (frozen) configuration.
        _validate_default_headers(self.default_headers)
        object.__setattr__(self, "default_headers", dict(self.default_headers))

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
            if self.api_port == DEFAULT_API_PORT:
                object.__setattr__(self, "api_port", self.single_port)
            if self.otlp_port == DEFAULT_OTLP_PORT:
                object.__setattr__(self, "otlp_port", self.single_port)

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

        if self.max_past_years != DEFAULT_MAX_PAST_YEARS:
            logger.warning(
                f"max_past_years is set to {self.max_past_years} (default: {DEFAULT_MAX_PAST_YEARS}). "
                "This setting allows timestamps older than the default limit. "
                "Please contact Arize support to enable custom timestamp limits for your account."
            )

    @property
    def cache_dir(self) -> str:
        """Return the path to the cache directory."""
        return str(Path(self.arize_directory) / "cache")

    @property
    def api_url(self) -> str:
        """Return the base API URL."""
        return _endpoint(self.api_scheme, self.api_host, port=self.api_port)

    @property
    def otlp_url(self) -> str:
        """Return the OTLP endpoint URL."""
        return _endpoint(
            self.otlp_scheme, self.otlp_host, "/v1", port=self.otlp_port
        )

    @property
    def files_url(self) -> str:
        """Return the files upload endpoint URL."""
        return _endpoint(
            self.api_scheme,
            self.api_host,
            "/v1/pandas_arrow",
            port=self.api_port,
        )

    @property
    def records_url(self) -> str:
        """Return the records logging endpoint URL."""
        return _endpoint(
            self.api_scheme, self.api_host, "/v1/log", port=self.api_port
        )

    @property
    def headers(self) -> dict[str, str]:
        """Return HTTP headers for API requests.

        User-supplied ``default_headers`` are included verbatim; the SDK's
        built-in headers are merged last and always win.
        """
        return {**self.default_headers, **_builtin_http_headers(self.api_key)}

    @property
    def headers_grpc(self) -> dict[str, str]:
        """Return headers for gRPC (grpc-gateway) requests.

        User-supplied ``default_headers`` are each prefixed with
        ``Grpc-Metadata-`` so grpc-gateway forwards them to the backend service;
        the SDK's built-in headers are merged last and always win.
        """
        prefixed = {
            f"Grpc-Metadata-{key}": value
            for key, value in self.default_headers.items()
        }
        return {**prefixed, **_builtin_grpc_headers(self.api_key)}

    @property
    def headers_flight(self) -> dict[str, str]:
        """Return headers for Apache Arrow Flight requests.

        User-supplied ``default_headers`` are included verbatim; the SDK's
        built-in headers are merged last and always win. The Flight client is
        responsible for byte-encoding these into the wire format.
        """
        return {**self.default_headers, **_builtin_flight_headers(self.api_key)}

    def __repr__(self) -> str:
        """Return a detailed string representation with masked sensitive fields."""
        lines = [f"{self.__class__.__name__}("]
        for f in fields(self):
            if not f.repr:
                continue
            val = getattr(self, f.name)
            if f.name == "default_headers":
                val = {
                    k: (_mask_secret(v, 6) if _is_sensitive_field(k) else v)
                    for k, v in val.items()
                }
            elif f.name == "proxy_url":
                val = _mask_proxy_url(val)
            elif _is_sensitive_field(f.name):
                val = _mask_secret(val, 6)
            lines.append(f"  {f.name}={val!r},")
        lines.append(")")
        return "\n".join(lines)
