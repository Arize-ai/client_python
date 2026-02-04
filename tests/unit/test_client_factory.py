"""Unit tests for src/arize/_client_factory.py."""

import threading
from unittest.mock import Mock, patch

import pytest

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
