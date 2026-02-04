"""Unit tests for src/arize/client.py."""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from arize.client import ArizeClient
from arize.regions import Region


@pytest.mark.unit
class TestArizeClientInit:
    """Tests for ArizeClient.__init__() parameter handling."""

    def test_init_with_api_key_only(self, test_api_key: str) -> None:
        """Initialize with only API key should succeed."""
        with patch("arize.client.SDKConfiguration") as mock_config_cls:
            mock_config_cls.return_value = Mock()
            ArizeClient(api_key=test_api_key)

            mock_config_cls.assert_called_once()
            assert "api_key" in mock_config_cls.call_args[1]
            assert mock_config_cls.call_args[1]["api_key"] == test_api_key

    @pytest.mark.parametrize(
        "param_name,param_value,config_key",
        [
            ("api_key", "test_api_key_12345", "api_key"),
            ("api_host", "custom.host.com", "api_host"),
            ("api_scheme", "http", "api_scheme"),
            ("otlp_host", "otlp.custom.com", "otlp_host"),
            ("otlp_scheme", "http", "otlp_scheme"),
            ("flight_host", "flight.custom.com", "flight_host"),
            ("flight_port", 8443, "flight_port"),
            ("flight_scheme", "grpc", "flight_scheme"),
            ("pyarrow_max_chunksize", 5000, "pyarrow_max_chunksize"),
            ("request_verify", False, "request_verify"),
            ("stream_max_workers", 4, "stream_max_workers"),
            ("stream_max_queue_bound", 1000, "stream_max_queue_bound"),
            ("max_http_payload_size_mb", 50.0, "max_http_payload_size_mb"),
            ("arize_directory", "/custom/dir", "arize_directory"),
            ("enable_caching", False, "enable_caching"),
            ("region", Region.EU_WEST_1A, "region"),
            ("single_host", "single.host.com", "single_host"),
            ("single_port", 9000, "single_port"),
            ("base_domain", "internal.example.com", "base_domain"),
        ],
    )
    def test_init_param_passed_to_config(
        self,
        test_api_key: str,
        param_name: str,
        param_value: object,
        config_key: str,
    ) -> None:
        """Individual parameters should be passed to SDKConfiguration."""
        with patch("arize.client.SDKConfiguration") as mock_config_cls:
            mock_config_cls.return_value = Mock()
            # Build kwargs dict - api_key is always needed, plus the param under test
            kwargs = {"api_key": test_api_key}
            # Only override api_key if that's the param being tested
            if param_name != "api_key":
                kwargs[param_name] = param_value
            else:
                kwargs["api_key"] = param_value

            ArizeClient(**kwargs)
            assert mock_config_cls.call_args[1][config_key] == param_value

    def test_init_none_params_not_passed(self, test_api_key: str) -> None:
        """None parameters should not be passed to SDKConfiguration."""
        with patch("arize.client.SDKConfiguration") as mock_config_cls:
            mock_config_cls.return_value = Mock()
            ArizeClient(
                api_key=test_api_key,
                api_host=None,
                region=None,
            )

            kwargs = mock_config_cls.call_args[1]
            assert "api_key" in kwargs
            assert "api_host" not in kwargs
            assert "region" not in kwargs


@pytest.mark.unit
class TestArizeClientEnvironmentVariables:
    """Tests for ArizeClient initialization with environment variables."""

    def test_client_uses_env_vars_when_no_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ArizeClient should read from env vars when no parameters provided."""
        # Set env vars
        monkeypatch.setenv("ARIZE_API_KEY", "env_key")
        monkeypatch.setenv("ARIZE_API_HOST", "env.arize.com")

        # Initialize with no parameters
        client = ArizeClient()

        # Verify config uses env vars
        assert client.sdk_config.api_key == "env_key"
        assert client.sdk_config.api_host == "env.arize.com"

    def test_explicit_params_override_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit parameters should override environment variables."""
        # Set env vars
        monkeypatch.setenv("ARIZE_API_KEY", "env_key")
        monkeypatch.setenv("ARIZE_API_HOST", "env.arize.com")

        # Initialize with explicit params
        client = ArizeClient(api_key="param_key", api_host="param.arize.com")

        # Verify params override env vars
        assert client.sdk_config.api_key == "param_key"
        assert client.sdk_config.api_host == "param.arize.com"

    def test_region_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ArizeClient should respect ARIZE_REGION env var."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv("ARIZE_REGION", "us-central-1a")

        client = ArizeClient()

        # Verify region affects endpoints
        assert "us-central-1a" in client.sdk_config.api_host

    def test_missing_api_key_env_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ArizeClient should raise error when ARIZE_API_KEY not set."""
        # Ensure ARIZE_API_KEY is not set
        monkeypatch.delenv("ARIZE_API_KEY", raising=False)

        # Should raise MissingAPIKeyError
        from arize.exceptions.auth import MissingAPIKeyError

        with pytest.raises(MissingAPIKeyError):
            ArizeClient()

    def test_partial_env_vars_with_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mix of env vars and params should work correctly."""
        monkeypatch.setenv("ARIZE_API_KEY", "env_key")
        # Don't set ARIZE_API_HOST - will use default

        # Override some params
        client = ArizeClient(api_scheme="http")

        # Verify: key from env, scheme from param, host from default
        assert client.sdk_config.api_key == "env_key"
        assert client.sdk_config.api_scheme == "http"
        assert client.sdk_config.api_host == "api.arize.com"  # default

    @pytest.mark.parametrize(
        "env_var_name,env_var_value,config_attr,expected_value",
        [
            ("ARIZE_ENABLE_CACHING", "false", "enable_caching", False),
            ("ARIZE_FLIGHT_PORT", "8443", "flight_port", 8443),
            ("ARIZE_API_SCHEME", "http", "api_scheme", "http"),
            (
                "ARIZE_OTLP_HOST",
                "custom-otlp.example.com",
                "otlp_host",
                "custom-otlp.example.com",
            ),
            ("ARIZE_OTLP_SCHEME", "http", "otlp_scheme", "http"),
            (
                "ARIZE_FLIGHT_HOST",
                "custom-flight.example.com",
                "flight_host",
                "custom-flight.example.com",
            ),
            ("ARIZE_FLIGHT_SCHEME", "grpc", "flight_scheme", "grpc"),
            (
                "ARIZE_SINGLE_HOST",
                "single.example.com",
                "single_host",
                "single.example.com",
            ),
            ("ARIZE_SINGLE_PORT", "9000", "single_port", 9000),
            ("ARIZE_MAX_CHUNKSIZE", "5000", "pyarrow_max_chunksize", 5000),
            ("ARIZE_REQUEST_VERIFY", "false", "request_verify", False),
            (
                "ARIZE_MAX_HTTP_PAYLOAD_SIZE_MB",
                "50.5",
                "max_http_payload_size_mb",
                50.5,
            ),
            (
                "ARIZE_DIRECTORY",
                "/custom/arize/dir",
                "arize_directory",
                "/custom/arize/dir",
            ),
            ("ARIZE_STREAM_MAX_WORKERS", "8", "stream_max_workers", 8),
            (
                "ARIZE_STREAM_MAX_QUEUE_BOUND",
                "2000",
                "stream_max_queue_bound",
                2000,
            ),
            (
                "ARIZE_BASE_DOMAIN",
                "env.internal.example.com",
                "base_domain",
                "env.internal.example.com",
            ),
        ],
    )
    def test_env_var_passed_to_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_var_name: str,
        env_var_value: str,
        config_attr: str,
        expected_value: object,
    ) -> None:
        """Environment variables should be parsed and passed to SDKConfiguration."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")
        monkeypatch.setenv(env_var_name, env_var_value)

        client = ArizeClient()

        assert getattr(client.sdk_config, config_attr) == expected_value


@pytest.mark.unit
class TestArizeClientProperties:
    """Tests for ArizeClient property access (lazy loading)."""

    @pytest.mark.parametrize(
        "property_name",
        ["datasets", "experiments", "ml", "projects", "spans"],
    )
    def test_property_lazy_loads(
        self, property_name: str, test_api_key: str
    ) -> None:
        """Accessing properties should trigger lazy loading."""
        with patch("arize.client.SDKConfiguration") as mock_config_cls:
            mock_config_cls.return_value = Mock()
            client = ArizeClient(api_key=test_api_key)

            with patch.object(
                client, "__getattr__", return_value=Mock()
            ) as mock_getattr:
                _ = getattr(client, property_name)
                mock_getattr.assert_called_once_with(property_name)


@pytest.mark.unit
class TestArizeClientRepr:
    """Tests for ArizeClient.__repr__() method."""

    def test_repr_structure(self, test_api_key: str) -> None:
        """Repr should have correct structure and include config."""
        with patch("arize.client.SDKConfiguration") as mock_config_cls:
            mock_config = Mock()
            mock_config.__repr__ = Mock(return_value="SDKConfiguration(...)")
            mock_config_cls.return_value = mock_config

            client = ArizeClient(api_key=test_api_key)
            repr_str = repr(client)

            # Check overall structure
            assert repr_str.startswith("ArizeClient(")
            assert repr_str.endswith(")")
            # Check that config is included
            assert "sdk_config=" in repr_str
            assert "subclients={" in repr_str

    def test_repr_includes_all_subclients(self, test_api_key: str) -> None:
        """Repr should include all subclients."""
        with patch("arize.client.SDKConfiguration") as mock_config_cls:
            mock_config = Mock()
            mock_config.__repr__ = Mock(return_value="SDKConfiguration(...)")
            mock_config_cls.return_value = mock_config

            client = ArizeClient(api_key=test_api_key)
            repr_str = repr(client)

            # Verify all subclients are mentioned
            for subclient in [
                "datasets",
                "experiments",
                "ml",
                "projects",
                "spans",
            ]:
                assert f"'{subclient}'" in repr_str


@pytest.mark.unit
class TestClearCache:
    """Tests for clear_cache() method."""

    def test_clear_cache_deletes_and_logs(
        self,
        tmp_path: Path,
        test_api_key: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Clear cache should delete existing cache directory and log operation."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "file.txt").write_text("test")

        with patch("arize.client.SDKConfiguration") as mock_config_cls:
            mock_config = Mock()
            mock_config.cache_dir = str(cache_dir)
            mock_config_cls.return_value = mock_config

            client = ArizeClient(api_key=test_api_key)
            caplog.set_level(logging.INFO)
            client.clear_cache()

            assert not cache_dir.exists()
            assert "Clearing cache directory" in caplog.text

    def test_clear_cache_recursive_deletion(
        self, tmp_path: Path, test_api_key: str
    ) -> None:
        """Clear cache should recursively delete all contents."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        subdir = cache_dir / "subdir"
        subdir.mkdir()
        (cache_dir / "file1.txt").write_text("test1")
        (subdir / "file2.txt").write_text("test2")

        with patch("arize.client.SDKConfiguration") as mock_config_cls:
            mock_config = Mock()
            mock_config.cache_dir = str(cache_dir)
            mock_config_cls.return_value = mock_config

            client = ArizeClient(api_key=test_api_key)
            client.clear_cache()

            # Everything should be deleted
            assert not cache_dir.exists()

    def test_clear_cache_tilde_expansion(
        self,
        tmp_path: Path,
        test_api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Clear cache should expand tilde in path."""
        # Create a temporary home directory
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        cache_dir = fake_home / ".arize" / "cache"
        cache_dir.mkdir(parents=True)

        monkeypatch.setenv("HOME", str(fake_home))

        with patch("arize.client.SDKConfiguration") as mock_config_cls:
            mock_config = Mock()
            mock_config.cache_dir = "~/.arize/cache"
            mock_config_cls.return_value = mock_config

            client = ArizeClient(api_key=test_api_key)
            client.clear_cache()

            # Cache dir should be deleted
            assert not cache_dir.exists()

    def test_clear_cache_error_cases(
        self,
        tmp_path: Path,
        test_api_key: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Clear cache should handle error cases: nonexistent dir and file path."""
        # Test nonexistent directory
        cache_dir = tmp_path / "nonexistent"

        with patch("arize.client.SDKConfiguration") as mock_config_cls:
            mock_config = Mock()
            mock_config.cache_dir = str(cache_dir)
            mock_config_cls.return_value = mock_config

            client = ArizeClient(api_key=test_api_key)
            caplog.set_level(logging.WARNING)

            with pytest.raises(NotADirectoryError):
                client.clear_cache()

            assert "does not exist" in caplog.text

        # Test file path instead of directory
        cache_file = tmp_path / "cache_file"
        cache_file.write_text("test")

        with patch("arize.client.SDKConfiguration") as mock_config_cls:
            mock_config = Mock()
            mock_config.cache_dir = str(cache_file)
            mock_config_cls.return_value = mock_config

            client = ArizeClient(api_key=test_api_key)
            caplog.clear()
            caplog.set_level(logging.ERROR)

            with pytest.raises(NotADirectoryError, match="Not a directory"):
                client.clear_cache()

            assert "not a directory" in caplog.text.lower()
