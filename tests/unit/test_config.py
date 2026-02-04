"""Unit tests for src/arize/config.py."""

import logging

import pytest

from arize.config import (
    SDKConfiguration,
    _endpoint,
    _env_bool,
    _env_float,
    _env_http_scheme,
    _env_int,
    _env_str,
    _is_sensitive_field,
    _mask_secret,
    _parse_bool,
)
from arize.exceptions.auth import MissingAPIKeyError
from arize.exceptions.config import MultipleEndpointOverridesError
from arize.regions import Region


@pytest.mark.unit
class TestIsSensitiveField:
    """Tests for _is_sensitive_field() function."""

    @pytest.mark.parametrize(
        "field_name,is_sensitive",
        [
            # Fields with 'key'
            ("api_key", True),
            ("secret_key", True),
            # Fields with 'token'
            ("auth_token", True),
            ("access_token", True),
            # Fields with 'secret'
            ("client_secret", True),
            ("api_secret", True),
            # Case insensitive matching
            ("API_KEY", True),
            ("Auth_Token", True),
            ("CLIENT_SECRET", True),
            # Non-sensitive fields
            ("api_host", False),
            ("port", False),
        ],
    )
    def test_is_sensitive_field(
        self, field_name: str, is_sensitive: bool
    ) -> None:
        """_is_sensitive_field should detect sensitive field names."""
        assert _is_sensitive_field(field_name) is is_sensitive
        assert _is_sensitive_field("scheme") is False


@pytest.mark.unit
class TestMaskSecret:
    """Tests for _mask_secret() function."""

    @pytest.mark.parametrize(
        "secret,n,expected",
        [
            ("my_secret_key", 4, "my_s***"),
            ("", 4, ""),
            ("secret123456", 6, "secret***"),
            ("x", 4, "x***"),
            ("very_long_secret_key_123456", 4, "very***"),
        ],
    )
    def test_mask_secret(self, secret: str, n: int, expected: str) -> None:
        """_mask_secret should mask secrets correctly with variable N parameter."""
        assert _mask_secret(secret, N=n) == expected


@pytest.mark.unit
class TestEndpoint:
    """Tests for _endpoint() function."""

    @pytest.mark.parametrize(
        "scheme,base,path,expected",
        [
            (
                "https",
                "api.example.com",
                "/v1/endpoint",
                "https://api.example.com/v1/endpoint",
            ),
            ("https", "api.example.com", "", "https://api.example.com"),
            (
                "https",
                "api.example.com",
                "v1/endpoint",
                "https://api.example.com/v1/endpoint",
            ),
            (
                "https",
                "api.example.com/",
                "/v1/endpoint",
                "https://api.example.com/v1/endpoint",
            ),
            ("http", "localhost:8080", "", "http://localhost:8080"),
        ],
    )
    def test_endpoint_construction(
        self, scheme: str, base: str, path: str, expected: str
    ) -> None:
        """_endpoint should construct URLs correctly with various input formats."""
        result = (
            _endpoint(scheme, base, path) if path else _endpoint(scheme, base)
        )
        assert result == expected


@pytest.mark.unit
class TestEnvHttpScheme:
    """Tests for _env_http_scheme() function."""

    def test_valid_https(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid HTTPS scheme should be accepted."""
        monkeypatch.setenv("TEST_SCHEME", "https")
        result = _env_http_scheme("TEST_SCHEME", "http")
        assert result == "https"

    def test_valid_http(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid HTTP scheme should be accepted."""
        monkeypatch.setenv("TEST_SCHEME", "http")
        result = _env_http_scheme("TEST_SCHEME", "https")
        assert result == "http"

    def test_invalid_scheme_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid scheme should raise ValueError."""
        monkeypatch.setenv("TEST_SCHEME", "ftp")
        with pytest.raises(ValueError, match="must be one of"):
            _env_http_scheme("TEST_SCHEME", "http")

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scheme matching should be case-insensitive."""
        monkeypatch.setenv("TEST_SCHEME", "HTTPS")
        result = _env_http_scheme("TEST_SCHEME", "http")
        assert result == "https"

    def test_uses_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use default when env var not set."""
        monkeypatch.delenv("TEST_SCHEME", raising=False)
        result = _env_http_scheme("TEST_SCHEME", "https")
        assert result == "https"


@pytest.mark.unit
class TestEnvStr:
    """Tests for _env_str() function."""

    def test_no_constraints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """String without constraints should be returned as-is."""
        monkeypatch.setenv("TEST_STR", "test_value")
        result = _env_str("TEST_STR", "default")
        assert result == "test_value"

    def test_min_length_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Min length constraint should be enforced."""
        monkeypatch.setenv("TEST_STR", "ab")
        with pytest.raises(ValueError, match="at least 5 characters"):
            _env_str("TEST_STR", "default", min_len=5)

    def test_max_length_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Max length constraint should be enforced."""
        monkeypatch.setenv("TEST_STR", "toolongvalue")
        with pytest.raises(ValueError, match="at most 5 characters"):
            _env_str("TEST_STR", "default", max_len=5)

    def test_whitespace_stripping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whitespace should be stripped from value."""
        monkeypatch.setenv("TEST_STR", "  value  ")
        result = _env_str("TEST_STR", "default")
        assert result == "value"

    def test_uses_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use default when env var not set."""
        monkeypatch.delenv("TEST_STR", raising=False)
        result = _env_str("TEST_STR", "default_value")
        assert result == "default_value"


@pytest.mark.unit
class TestEnvInt:
    """Tests for _env_int() function."""

    def test_parses_valid_integer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Valid integer should be parsed."""
        monkeypatch.setenv("TEST_INT", "42")
        result = _env_int("TEST_INT", 0)
        assert result == 42

    def test_invalid_format_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid integer format should raise ValueError."""
        monkeypatch.setenv("TEST_INT", "not_a_number")
        with pytest.raises(ValueError, match="must be an int"):
            _env_int("TEST_INT", 0)

    @pytest.mark.parametrize(
        "value,min_val,max_val,should_raise,expected_result",
        [
            ("42", None, None, False, 42),
            ("5", 10, None, True, None),
            ("100", None, 50, True, None),
            ("25", 10, 50, False, 25),
        ],
    )
    def test_env_int_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: str,
        min_val: int | None,
        max_val: int | None,
        should_raise: bool,
        expected_result: int | None,
    ) -> None:
        """_env_int should validate min/max constraints correctly."""
        monkeypatch.setenv("TEST_INT", value)
        if should_raise:
            with pytest.raises(ValueError):
                _env_int("TEST_INT", 0, min_val=min_val, max_val=max_val)
        else:
            assert (
                _env_int("TEST_INT", 0, min_val=min_val, max_val=max_val)
                == expected_result
            )

    def test_negative_numbers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Negative integers should be parsed."""
        monkeypatch.setenv("TEST_INT", "-42")
        result = _env_int("TEST_INT", 0)
        assert result == -42


@pytest.mark.unit
class TestEnvFloat:
    """Tests for _env_float() function."""

    def test_parses_valid_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid float should be parsed."""
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        result = _env_float("TEST_FLOAT", 0.0)
        assert result == 3.14

    def test_invalid_format_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid float format should raise ValueError."""
        monkeypatch.setenv("TEST_FLOAT", "not_a_number")
        with pytest.raises(ValueError, match="must be a float"):
            _env_float("TEST_FLOAT", 0.0)

    @pytest.mark.parametrize(
        "value,min_val,max_val,should_raise,expected_result",
        [
            ("42.5", None, None, False, 42.5),
            ("5.5", 10.0, None, True, None),
            ("100.5", None, 50.0, True, None),
            ("25.5", 10.0, 50.0, False, 25.5),
        ],
    )
    def test_env_float_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: str,
        min_val: float | None,
        max_val: float | None,
        should_raise: bool,
        expected_result: float | None,
    ) -> None:
        """_env_float should validate min/max constraints correctly."""
        monkeypatch.setenv("TEST_FLOAT", value)
        if should_raise:
            with pytest.raises(ValueError):
                _env_float("TEST_FLOAT", 0.0, min_val=min_val, max_val=max_val)
        else:
            assert (
                _env_float("TEST_FLOAT", 0.0, min_val=min_val, max_val=max_val)
                == expected_result
            )

    def test_scientific_notation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scientific notation should be parsed."""
        monkeypatch.setenv("TEST_FLOAT", "1.5e10")
        result = _env_float("TEST_FLOAT", 0.0)
        assert result == 1.5e10


@pytest.mark.unit
class TestEnvBool:
    """Tests for _env_bool() function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            # Truthy values
            ("1", True),
            ("true", True),
            ("True", True),
            ("yes", True),
            ("YES", True),
            ("on", True),
            ("ON", True),
            # Falsy values
            ("0", False),
            ("false", False),
            ("False", False),
            ("no", False),
            ("off", False),
            ("", False),
        ],
    )
    def test_env_bool_parsing(
        self, monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
    ) -> None:
        """_env_bool should correctly parse truthy and falsy values."""
        monkeypatch.setenv("TEST_BOOL", value)
        assert _env_bool("TEST_BOOL", not expected) is expected

    def test_uses_default_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use default when env var not set."""
        monkeypatch.delenv("TEST_BOOL", raising=False)
        assert _env_bool("TEST_BOOL", True) is True
        assert _env_bool("TEST_BOOL", False) is False


@pytest.mark.unit
class TestParseBool:
    """Tests for _parse_bool() function."""

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            # Boolean inputs
            (True, True),
            (False, False),
            # Truthy strings
            ("1", True),
            ("true", True),
            ("yes", True),
            ("on", True),
            # Falsy strings
            ("0", False),
            ("false", False),
            ("no", False),
            ("off", False),
            # Edge cases
            (None, False),
            ("", False),
        ],
    )
    def test_parse_bool(
        self, input_value: bool | str | None, expected: bool
    ) -> None:
        """_parse_bool should handle all input types correctly."""
        assert _parse_bool(input_value) is expected


@pytest.mark.unit
class TestSDKConfiguration:
    """Tests for SDKConfiguration class."""

    def test_init_with_all_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Initialization with minimal params should use defaults."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key_12345")
        config = SDKConfiguration()

        assert config.api_key == "test_key_12345"
        assert config.api_host == "api.arize.com"
        assert config.api_scheme == "https"

    def test_init_with_explicit_values(self) -> None:
        """Initialization with explicit values should override defaults."""
        config = SDKConfiguration(
            api_key="explicit_key",
            api_host="custom.host.com",
            api_scheme="http",
        )

        assert config.api_key == "explicit_key"
        assert config.api_host == "custom.host.com"
        assert config.api_scheme == "http"

    def test_init_missing_api_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing API key should raise MissingAPIKeyError."""
        monkeypatch.delenv("ARIZE_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError):
            SDKConfiguration()

    def test_region_overrides_all_endpoints(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Region should override all endpoint settings."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration(
            api_host="custom.host.com",
            region=Region.US_CENTRAL_1A,
        )

        caplog.set_level(logging.INFO)
        # The post_init already ran, check the values
        assert "us-central-1a" in config.api_host
        assert "us-central-1a" in config.otlp_host
        assert "us-central-1a" in config.flight_host

    def test_single_host_overrides_hosts(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Single host should override individual host settings."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration(
            single_host="single.host.com",
        )

        assert config.api_host == "single.host.com"
        assert config.otlp_host == "single.host.com"
        assert config.flight_host == "single.host.com"

    def test_single_port_overrides_flight_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single port should override flight port."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration(
            single_port=9000,
        )

        assert config.flight_port == 9000

    def test_multiple_overrides_raises_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Multiple endpoint overrides should raise MultipleEndpointOverridesError."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        caplog.set_level(logging.ERROR)

        with pytest.raises(
            MultipleEndpointOverridesError,
            match="Multiple endpoint override options",
        ):
            SDKConfiguration(
                single_host="single.host.com",
                region=Region.EU_WEST_1A,
            )

        # Should log error before raising
        assert "Multiple endpoint override options" in caplog.text

    def test_override_logging(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Override operations should log info messages."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        caplog.set_level(logging.INFO)

        SDKConfiguration(
            single_host="single.host.com",
        )

        assert "Single host" in caplog.text
        assert "overriding hosts configuration" in caplog.text

    def test_cache_dir_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cache dir property should return correct path."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration(arize_directory="/custom/dir")

        assert config.cache_dir == "/custom/dir/cache"

    def test_api_url_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """API URL property should construct correct URL."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration(
            api_scheme="https",
            api_host="api.example.com",
        )

        assert config.api_url == "https://api.example.com"

    def test_otlp_url_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OTLP URL property should construct correct URL with path."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration(
            otlp_scheme="https",
            otlp_host="otlp.example.com",
        )

        assert config.otlp_url == "https://otlp.example.com/v1"

    def test_files_url_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Files URL property should construct correct URL with path."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration()

        assert "/v1/pandas_arrow" in config.files_url

    def test_records_url_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Records URL property should construct correct URL with path."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration()

        assert "/v1/log" in config.records_url

    def test_headers_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Headers property should include required headers."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key_12345")
        config = SDKConfiguration()

        headers = config.headers
        assert headers["authorization"] == "test_key_12345"
        assert headers["sdk-language"] == "python"
        assert "language-version" in headers
        assert "sdk-version" in headers

    def test_headers_grpc_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GRPC headers property should include required headers."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key_12345")
        config = SDKConfiguration()

        headers = config.headers_grpc
        assert headers["authorization"] == "test_key_12345"
        assert headers["Grpc-Metadata-sdk-language"] == "python"
        assert "Grpc-Metadata-language-version" in headers
        assert "Grpc-Metadata-sdk-version" in headers

    def test_repr_masks_sensitive_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repr should mask sensitive fields."""
        monkeypatch.setenv("ARIZE_API_KEY", "secret_key_12345")
        config = SDKConfiguration()

        repr_str = repr(config)
        assert "secret***" in repr_str
        assert "secret_key_12345" not in repr_str

    def test_repr_shows_non_sensitive_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repr should show non-sensitive fields."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration(api_host="custom.host.com")

        repr_str = repr(config)
        assert "custom.host.com" in repr_str

    def test_repr_formatting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repr should have correct formatting."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration()

        repr_str = repr(config)
        assert repr_str.startswith("SDKConfiguration(")
        assert repr_str.endswith(")")

    def test_frozen_dataclass_immutable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Frozen dataclass should be immutable."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration()

        # FrozenInstanceError (subclass of AttributeError) in Python 3.11+
        with pytest.raises((AttributeError, TypeError)):
            config.api_host = "new_host"  # type: ignore

    def test_path_tilde_expansion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cache dir should handle tilde expansion."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration(arize_directory="~/.arize")

        # Cache dir should contain tilde (not expanded by config itself)
        assert "cache" in config.cache_dir

    def test_default_factory_functions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default factory functions should be called correctly."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.delenv("ARIZE_API_HOST", raising=False)

        config = SDKConfiguration()
        # Should use default from factory
        assert config.api_host == "api.arize.com"

    def test_field_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Field validation should work through env vars."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv("ARIZE_FLIGHT_PORT", "8443")

        # Should not raise - port is within valid range (1-65535)
        config = SDKConfiguration()
        assert config.flight_port == 8443

    def test_base_domain_api_url_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """API URL property should use base_domain when provided."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration(base_domain="internal.example.com")

        assert config.api_url == "https://api.internal.example.com"

    def test_base_domain_otlp_url_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OTLP URL property should use base_domain when provided."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration(base_domain="internal.example.com")

        assert config.otlp_url == "https://otlp.internal.example.com/v1"

    def test_base_domain_files_url_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Files URL property should use base_domain when provided."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration(base_domain="internal.example.com")

        assert (
            config.files_url
            == "https://api.internal.example.com/v1/pandas_arrow"
        )

    def test_base_domain_records_url_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Records URL property should use base_domain when provided."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        config = SDKConfiguration(base_domain="internal.example.com")

        assert config.records_url == "https://api.internal.example.com/v1/log"


@pytest.mark.unit
class TestConfigurationPrecedence:
    """Tests for configuration precedence rules."""

    def test_region_with_single_host_raises_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Region with single_host should raise MultipleEndpointOverridesError."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        caplog.set_level(logging.ERROR)

        with pytest.raises(
            MultipleEndpointOverridesError,
            match="Multiple endpoint override options",
        ):
            SDKConfiguration(
                single_host="custom.host.com",
                region=Region.US_CENTRAL_1A,
            )

        # Should log error before raising
        assert "Multiple endpoint override options" in caplog.text

    def test_explicit_param_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit parameter should override environment variable."""
        monkeypatch.setenv("ARIZE_API_KEY", "env_key")
        monkeypatch.setenv("ARIZE_API_HOST", "env.host.com")

        config = SDKConfiguration(api_host="param.host.com")

        # Parameter should win over env var
        assert config.api_host == "param.host.com"

    def test_env_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment variable should override default value."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv("ARIZE_API_HOST", "env.custom.com")

        config = SDKConfiguration()

        assert config.api_host == "env.custom.com"

    def test_default_used_when_nothing_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default should be used when neither param nor env set."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.delenv("ARIZE_API_HOST", raising=False)

        config = SDKConfiguration()

        assert config.api_host == "api.arize.com"

    def test_base_domain_with_region_from_env_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """base_domain and region both from env vars should raise error."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv("ARIZE_BASE_DOMAIN", "internal.com")
        monkeypatch.setenv("ARIZE_REGION", "us-central-1a")

        with pytest.raises(
            MultipleEndpointOverridesError,
            match="Multiple endpoint override options",
        ):
            SDKConfiguration()

    def test_base_domain_with_single_host_from_env_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """base_domain and single_host both from env vars should raise error."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv("ARIZE_BASE_DOMAIN", "internal.com")
        monkeypatch.setenv("ARIZE_SINGLE_HOST", "single.host.com")

        with pytest.raises(
            MultipleEndpointOverridesError,
            match="Multiple endpoint override options",
        ):
            SDKConfiguration()

    def test_mixed_env_and_param_overrides_raise_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mixed env vars and params with conflicts should raise error."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv("ARIZE_BASE_DOMAIN", "env.internal.com")

        with pytest.raises(
            MultipleEndpointOverridesError,
            match="Multiple endpoint override options",
        ):
            SDKConfiguration(region=Region.EU_WEST_1A)


@pytest.mark.unit
class TestEnvironmentVariableParsing:
    """Tests for environment variable parsing."""

    def test_invalid_int_env_var_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid integer environment variable should raise ValueError."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv("ARIZE_FLIGHT_PORT", "not_a_number")

        with pytest.raises(ValueError, match="must be an int"):
            SDKConfiguration()

    def test_invalid_scheme_env_var_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid scheme environment variable should raise ValueError."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv("ARIZE_API_SCHEME", "ftp")

        with pytest.raises(ValueError, match="must be one of"):
            SDKConfiguration()

    def test_port_out_of_range_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Port out of valid range should raise ValueError."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv("ARIZE_FLIGHT_PORT", "99999")

        with pytest.raises(ValueError, match="at most 65535"):
            SDKConfiguration()


@pytest.mark.unit
class TestConfigurationValidation:
    """Tests for configuration validation."""

    @pytest.mark.parametrize(
        "api_key_value,should_set_env",
        [
            (None, False),  # Missing
            ("", True),  # Empty
            ("   ", True),  # Whitespace
        ],
    )
    def test_invalid_api_key_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        api_key_value: str | None,
        should_set_env: bool,
    ) -> None:
        """Missing, empty, or whitespace-only API keys should raise MissingAPIKeyError."""
        if should_set_env:
            monkeypatch.setenv("ARIZE_API_KEY", api_key_value)
        else:
            monkeypatch.delenv("ARIZE_API_KEY", raising=False)

        with pytest.raises(MissingAPIKeyError):
            SDKConfiguration()


@pytest.mark.unit
class TestClientWithConfiguration:
    """Tests for ArizeClient with various configurations."""

    def test_client_with_region_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client should respect ARIZE_REGION environment variable."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv("ARIZE_REGION", "eu-west-1a")

        from arize import ArizeClient

        client = ArizeClient()

        assert "eu-west-1a" in client.sdk_config.api_host

    def test_client_with_custom_directory(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client should use custom arize_directory."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        from arize import ArizeClient

        custom_dir = str(tmp_path)
        client = ArizeClient(arize_directory=custom_dir)

        assert client.sdk_config.arize_directory == custom_dir
        assert custom_dir in client.sdk_config.cache_dir

    def test_client_with_disabled_caching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client should respect enable_caching=False."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        from arize import ArizeClient

        client = ArizeClient(enable_caching=False)

        assert client.sdk_config.enable_caching is False

    def test_client_with_custom_ports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client should use custom port configurations."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        from arize import ArizeClient

        client = ArizeClient(flight_port=9443)

        assert client.sdk_config.flight_port == 9443

    def test_client_with_base_domain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ArizeClient should work end-to-end with base_domain."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        from arize import ArizeClient

        client = ArizeClient(base_domain="internal.example.com")

        # Verify config is set correctly
        assert client.sdk_config.api_host == "api.internal.example.com"
        assert client.sdk_config.otlp_host == "otlp.internal.example.com"
        assert client.sdk_config.flight_host == "flight.internal.example.com"

        # Verify URL properties
        assert client.sdk_config.api_url == "https://api.internal.example.com"
        assert (
            client.sdk_config.otlp_url == "https://otlp.internal.example.com/v1"
        )

    def test_client_with_base_domain_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ArizeClient should respect ARIZE_BASE_DOMAIN env var."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv("ARIZE_BASE_DOMAIN", "env.internal.com")

        from arize import ArizeClient

        client = ArizeClient()

        # Verify config uses env var
        assert client.sdk_config.api_host == "api.env.internal.com"
        assert client.sdk_config.otlp_host == "otlp.env.internal.com"
        assert client.sdk_config.flight_host == "flight.env.internal.com"

    def test_client_with_single_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client should work with single_host parameter."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        from arize import ArizeClient

        client = ArizeClient(single_host="custom.host.com")

        assert client.sdk_config.api_host == "custom.host.com"
        assert client.sdk_config.otlp_host == "custom.host.com"
        assert client.sdk_config.flight_host == "custom.host.com"

    def test_client_with_single_host_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client should respect ARIZE_SINGLE_HOST env var."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv("ARIZE_SINGLE_HOST", "env.host.com")

        from arize import ArizeClient

        client = ArizeClient()

        assert client.sdk_config.api_host == "env.host.com"
        assert client.sdk_config.otlp_host == "env.host.com"
        assert client.sdk_config.flight_host == "env.host.com"


@pytest.mark.unit
class TestMultipleEndpointOverridesError:
    """Tests for MultipleEndpointOverridesError validation."""

    def test_error_logs_before_raising(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Error should be logged before raising exception."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        caplog.set_level(logging.ERROR)

        with pytest.raises(MultipleEndpointOverridesError):
            SDKConfiguration(
                base_domain="internal.example.com",
                region=Region.US_CENTRAL_1A,
            )

        # Should log error before exception
        assert caplog.records
        assert any(record.levelname == "ERROR" for record in caplog.records)
        assert any(
            "Multiple endpoint override options" in record.message
            for record in caplog.records
        )

    @pytest.mark.parametrize(
        "kwargs,expected_in_error",
        [
            (
                {"base_domain": "x.com", "region": Region.US_CENTRAL_1A},
                ["base_domain", "region"],
            ),
            (
                {"base_domain": "x.com", "single_host": "y.com"},
                ["base_domain", "single_host"],
            ),
            (
                {"base_domain": "x.com", "single_port": 9000},
                ["base_domain", "single_port"],
            ),
            (
                {"region": Region.EU_WEST_1A, "single_host": "y.com"},
                ["region", "single_host"],
            ),
            (
                {"region": Region.CA_CENTRAL_1A, "single_port": 9000},
                ["region", "single_port"],
            ),
            # Note: single_host + single_port together is ALLOWED (they're one override type)
            (
                {
                    "base_domain": "x.com",
                    "region": Region.US_EAST_1B,
                    "single_host": "y.com",
                },
                ["base_domain", "region", "single_host"],
            ),
        ],
    )
    def test_all_conflict_combinations(
        self,
        monkeypatch: pytest.MonkeyPatch,
        kwargs: dict,
        expected_in_error: list[str],
    ) -> None:
        """All conflict combinations should raise error with appropriate message."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        with pytest.raises(MultipleEndpointOverridesError) as exc_info:
            SDKConfiguration(**kwargs)

        error_msg = str(exc_info.value)
        # Verify error message mentions the conflicting options
        for expected in expected_in_error:
            assert (
                expected in error_msg or expected.replace("_", " ") in error_msg
            )
