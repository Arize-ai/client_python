"""Unit tests for src/arize/_client_factory.py."""

import base64
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock, patch

import pytest
from urllib3.exceptions import ProxyError

from arize._client_factory import GeneratedClientFactory


@pytest.fixture
def factory(mock_sdk_config: Mock) -> GeneratedClientFactory:
    """Create a GeneratedClientFactory instance for testing."""
    return GeneratedClientFactory(mock_sdk_config)


@pytest.mark.unit
class TestGeneratedClientFactoryInit:
    """Tests for GeneratedClientFactory initialization."""

    def test_init_stores_config(self, mock_sdk_config: Mock) -> None:
        """Factory should store the SDK configuration."""
        factory = GeneratedClientFactory(mock_sdk_config)
        assert factory._sdk_config is mock_sdk_config

    def test_init_creates_lock_and_null_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Factory should initialize with a lock and null client."""
        factory = GeneratedClientFactory(mock_sdk_config)
        assert isinstance(factory._lock, type(threading.Lock()))
        assert factory._client is None


@pytest.mark.unit
class TestGetClient:
    """Tests for get_client() method."""

    def test_first_call_creates_client(
        self, factory: GeneratedClientFactory, mock_sdk_config: Mock
    ) -> None:
        """First call should create and return a new client."""
        with (
            patch(
                "arize._generated.api_client.Configuration"
            ) as mock_config_cls,
            patch("arize._generated.api_client.ApiClient") as mock_client_cls,
        ):
            mock_config_instance = Mock()
            mock_client_instance = Mock()
            mock_config_cls.return_value = mock_config_instance
            mock_client_cls.return_value = mock_client_instance

            result = factory.get_client()

            assert result is mock_client_instance
            mock_config_cls.assert_called_once_with(
                host=mock_sdk_config.api_url
            )
            mock_client_cls.assert_called_once_with(mock_config_instance)

    def test_subsequent_calls_return_cached(
        self, factory: GeneratedClientFactory
    ) -> None:
        """Subsequent calls should return the cached client instance."""
        with (
            patch(
                "arize._generated.api_client.Configuration"
            ) as mock_config_cls,
            patch("arize._generated.api_client.ApiClient") as mock_client_cls,
        ):
            mock_config_instance = Mock()
            mock_client_instance = Mock()
            mock_config_cls.return_value = mock_config_instance
            mock_client_cls.return_value = mock_client_instance

            first_result = factory.get_client()
            second_result = factory.get_client()

            assert first_result is second_result
            # Configuration and ApiClient should only be called once
            assert mock_config_cls.call_count == 1
            assert mock_client_cls.call_count == 1

    def test_lazy_import_executed_once(
        self, factory: GeneratedClientFactory
    ) -> None:
        """Lazy import should only execute once even with multiple calls."""
        with (
            patch(
                "arize._generated.api_client.Configuration"
            ) as mock_config_cls,
            patch("arize._generated.api_client.ApiClient") as mock_client_cls,
        ):
            mock_config_instance = Mock()
            mock_client_instance = Mock()
            mock_config_cls.return_value = mock_config_instance
            mock_client_cls.return_value = mock_client_instance

            factory.get_client()
            factory.get_client()
            factory.get_client()

            # Should only create configuration and client once
            assert mock_config_cls.call_count == 1
            assert mock_client_cls.call_count == 1

    def test_configuration_uses_sdk_config(
        self, factory: GeneratedClientFactory, mock_sdk_config: Mock
    ) -> None:
        """Configuration should use the SDK config's api_url."""
        with (
            patch(
                "arize._generated.api_client.Configuration"
            ) as mock_config_cls,
            patch("arize._generated.api_client.ApiClient") as mock_client_cls,
        ):
            mock_config_instance = Mock()
            mock_client_instance = Mock()
            mock_config_cls.return_value = mock_config_instance
            mock_client_cls.return_value = mock_client_instance

            factory.get_client()

            mock_config_cls.assert_called_once_with(
                host=mock_sdk_config.api_url
            )

    def test_api_key_passed_to_config(
        self, factory: GeneratedClientFactory, mock_sdk_config: Mock
    ) -> None:
        """API key should be set on the configuration if present."""
        with (
            patch(
                "arize._generated.api_client.Configuration"
            ) as mock_config_cls,
            patch("arize._generated.api_client.ApiClient") as mock_client_cls,
        ):
            mock_config_instance = Mock()
            mock_client_instance = Mock()
            mock_config_cls.return_value = mock_config_instance
            mock_client_cls.return_value = mock_client_instance

            factory.get_client()

            assert mock_config_instance.access_token == mock_sdk_config.api_key

    @pytest.mark.parametrize("verify", [True, False])
    def test_verify_ssl_set_from_request_verify(
        self,
        mock_sdk_config: Mock,
        verify: bool,
    ) -> None:
        """verify_ssl on the generated Configuration should mirror request_verify."""
        mock_sdk_config.request_verify = verify
        factory = GeneratedClientFactory(mock_sdk_config)

        with (
            patch(
                "arize._generated.api_client.Configuration"
            ) as mock_config_cls,
            patch("arize._generated.api_client.ApiClient"),
        ):
            mock_config_instance = Mock()
            mock_config_cls.return_value = mock_config_instance

            factory.get_client()

            assert mock_config_instance.verify_ssl == verify

    def test_api_key_not_set_if_none(self, mock_sdk_config: Mock) -> None:
        """API key should not be set on configuration if it's None."""
        mock_sdk_config.api_key = None
        factory = GeneratedClientFactory(mock_sdk_config)

        with (
            patch(
                "arize._generated.api_client.Configuration"
            ) as mock_config_cls,
            patch("arize._generated.api_client.ApiClient") as mock_client_cls,
        ):
            mock_config_instance = Mock(spec=["host"])
            mock_client_instance = Mock()
            mock_config_cls.return_value = mock_config_instance
            mock_client_cls.return_value = mock_client_instance

            factory.get_client()

            # access_token should not have been set when api_key is None
            # Since we used spec, accessing access_token should not exist
            with pytest.raises(AttributeError):
                _ = mock_config_instance.access_token

    def test_thread_safety_single_instance(
        self, factory: GeneratedClientFactory
    ) -> None:
        """Multiple threads should all receive the same client instance."""
        with (
            patch(
                "arize._generated.api_client.Configuration"
            ) as mock_config_cls,
            patch("arize._generated.api_client.ApiClient") as mock_client_cls,
        ):
            mock_config_instance = Mock()
            mock_client_instance = Mock()
            mock_config_cls.return_value = mock_config_instance
            mock_client_cls.return_value = mock_client_instance

            results: list = []

            def get_and_store() -> None:
                results.append(factory.get_client())

            threads = [
                threading.Thread(target=get_and_store) for _ in range(10)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All threads should get the same instance
            assert len(results) == 10
            assert all(r is results[0] for r in results)
            # Client should only be created once
            assert mock_client_cls.call_count == 1

    def test_fast_path_no_lock_when_cached(
        self, factory: GeneratedClientFactory
    ) -> None:
        """Fast path should return cached client without acquiring lock."""
        # Manually set a cached client to test the fast path
        mock_client = Mock()
        factory._client = mock_client

        result = factory.get_client()

        # Should return cached client without needing to import or lock
        assert result is mock_client


@pytest.mark.unit
@pytest.mark.parametrize(
    ("proxy_url", "credentials"),
    [
        ("http://user:password@localhost:8080", "user:password"),
        ("https://user:password@localhost:8080", "user:password"),
        ("http://user%40corp:p%3Aa%2Bss@localhost:8080", "user@corp:p:a+ss"),
        ("http://user+name:p+ss@localhost:8080", "user+name:p+ss"),
        (
            "http://user:%E2%82%AC%E5%AF%86%E7%A2%BC@localhost:8080",
            "user:€密碼",
        ),
        ("http://user:caf%C3%A9@localhost:8080", "user:café"),
        ("http://user@localhost:8080", "user:"),
        ("http://:password@localhost:8080", ":password"),
        ("http://localhost:8080", None),
        ("", None),
    ],
)
def test_proxy_credentials_stay_out_of_origin_headers(
    mock_sdk_config: Mock, proxy_url: str, credentials: str | None
) -> None:
    """The real REST transport receives proxy-only, URL-decoded credentials."""
    mock_sdk_config.proxy_url = proxy_url
    client = GeneratedClientFactory(mock_sdk_config).get_client()
    try:
        expected = (
            {
                "proxy-authorization": "Basic "
                + base64.b64encode(credentials.encode()).decode()
            }
            if credentials is not None
            else None
        )
        assert client.configuration.proxy_headers == expected
        if proxy_url:
            assert client.rest_client.pool_manager.proxy_headers == (
                expected or {}
            )
        assert "proxy-authorization" not in {
            k.lower() for k in client.default_headers
        }
        assert client.configuration.access_token == mock_sdk_config.api_key
    finally:
        client.rest_client.pool_manager.clear()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("proxy_url", "expected_url"),
    [
        ("http://user:p%40ss@localhost:8080", "http://localhost:8080"),
        ("https://user:password@localhost", "https://localhost"),
        ("http://user:password@[::1]:8080", "http://[::1]:8080"),
        ("https://user:password@[::1]", "https://[::1]"),
    ],
)
def test_http_proxy_url_excludes_credentials(
    mock_sdk_config: Mock, proxy_url: str, expected_url: str
) -> None:
    """Proxy URLs omit credentials while retaining the configured address."""
    mock_sdk_config.proxy_url = proxy_url
    client = GeneratedClientFactory(mock_sdk_config).get_client()
    try:
        assert client.configuration.proxy == expected_url
        assert client.rest_client.pool_manager.proxy.auth is None
        expected_transport_url = (
            f"{expected_url}:443"
            if expected_url.startswith("https:")
            else expected_url
        )
        assert (
            client.rest_client.pool_manager.proxy.url == expected_transport_url
        )
    finally:
        client.rest_client.pool_manager.clear()


@pytest.mark.unit
def test_socks_proxy_keeps_url_authentication(mock_sdk_config: Mock) -> None:
    """SOCKS authentication remains in its URL, without HTTP proxy headers."""
    mock_sdk_config.proxy_url = "socks5h://user:password@localhost:1080"
    with patch("arize._generated.api_client.ApiClient") as constructor:
        GeneratedClientFactory(mock_sdk_config).get_client()
    config = constructor.call_args.args[0]
    assert config.proxy == mock_sdk_config.proxy_url
    assert config.proxy_headers is None


@pytest.mark.unit
def test_authenticated_proxy_connect(mock_sdk_config: Mock) -> None:
    """CONNECT carries the proxy credential without forwarding the API key."""
    received = []

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_CONNECT(self) -> None:
            received.append((self.path, dict(self.headers)))
            self.send_error(502)

        def log_message(self, format: str, *args: object) -> None:
            pass

    with HTTPServer(("127.0.0.1", 0), ProxyHandler) as server:
        server.timeout = 2
        worker = threading.Thread(target=server.handle_request, daemon=True)
        worker.start()
        mock_sdk_config.proxy_url = (
            f"http://user:p%40ss@127.0.0.1:{server.server_port}"
        )
        client = GeneratedClientFactory(mock_sdk_config).get_client()
        try:
            with pytest.raises(ProxyError):
                client.rest_client.pool_manager.request(
                    "GET",
                    "https://origin.invalid/spans",
                    headers={"Authorization": "Bearer dummy-api-key"},
                    retries=False,
                    timeout=1,
                )
        finally:
            client.rest_client.pool_manager.clear()
            worker.join(timeout=3)
        assert not worker.is_alive()
    assert len(received) == 1
    target, headers = received[0]
    assert target == "origin.invalid:443"
    headers = {key.lower(): value for key, value in headers.items()}
    assert headers["proxy-authorization"] == "Basic dXNlcjpwQHNz"
    assert "authorization" not in headers
