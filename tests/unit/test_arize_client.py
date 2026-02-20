"""Unit tests for client lifecycle, lazy loading, and factory.

These tests validate that ArizeClient, LazySubclientsMixin, and GeneratedClientFactory
work correctly in isolated scenarios (unit tests without external service dependencies).
"""

import threading
from pathlib import Path

import pytest

from arize import ArizeClient
from arize.config import SDKConfiguration
from arize.regions import Region


@pytest.mark.unit
class TestClientInitialization:
    """Unit tests for ArizeClient initialization."""

    def test_client_initialization_with_real_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client should initialize with real SDKConfiguration."""
        monkeypatch.setenv("ARIZE_API_KEY", "integration_test_key_12345")

        client = ArizeClient()

        # Verify client has real config
        assert isinstance(client.sdk_config, SDKConfiguration)
        assert client.sdk_config.api_key == "integration_test_key_12345"
        assert client.sdk_config.api_host == "api.arize.com"

    def test_client_with_all_parameters(self) -> None:
        """Client should accept and use all configuration parameters."""
        client = ArizeClient(
            api_key="test_key_12345",
            api_host="custom.api.com",
            api_scheme="http",
            otlp_host="custom.otlp.com",
            flight_host="custom.flight.com",
            flight_port=8443,
            arize_directory="/tmp/test_arize",
            enable_caching=False,
        )

        config = client.sdk_config
        assert config.api_key == "test_key_12345"
        assert config.api_host == "custom.api.com"
        assert config.api_scheme == "http"
        assert config.otlp_host == "custom.otlp.com"
        assert config.flight_host == "custom.flight.com"
        assert config.flight_port == 8443
        assert config.arize_directory == "/tmp/test_arize"
        assert config.enable_caching is False

    def test_client_with_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client should apply region configuration correctly."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient(region=Region.US_CENTRAL_1A)

        # Region should override default hosts
        assert "us-central-1a" in client.sdk_config.api_host
        assert "us-central-1a" in client.sdk_config.otlp_host
        assert "us-central-1a" in client.sdk_config.flight_host

    def test_client_repr_includes_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Client repr should include configuration details."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key_12345")

        client = ArizeClient()
        repr_str = repr(client)

        assert "ArizeClient(" in repr_str
        assert "sdk_config=" in repr_str
        assert "subclients=" in repr_str
        # API key should be masked
        assert "test_k***" in repr_str or "***" in repr_str


@pytest.mark.unit
class TestLazyLoadingRealSubclients:
    """Unit tests for lazy loading with real subclients."""

    @pytest.mark.parametrize(
        "subclient_name,expected_class_path",
        [
            ("datasets", "arize.datasets.client.DatasetsClient"),
            ("experiments", "arize.experiments.client.ExperimentsClient"),
            ("ml", "arize.ml.client.MLModelsClient"),
            ("projects", "arize.projects.client.ProjectsClient"),
            ("spans", "arize.spans.client.SpansClient"),
            (
                "annotation_configs",
                "arize.annotation_configs.client.AnnotationConfigsClient",
            ),
        ],
    )
    def test_lazy_loading_subclients(
        self,
        monkeypatch: pytest.MonkeyPatch,
        subclient_name: str,
        expected_class_path: str,
    ) -> None:
        """Accessing subclient should lazy load real instance."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient()

        # Before access, not loaded
        assert subclient_name not in client._lazy_cache

        # Access triggers lazy loading
        subclient = getattr(client, subclient_name)

        # After access, should be cached and be a real instance
        assert subclient_name in client._lazy_cache
        assert subclient is not None
        assert client._lazy_cache[subclient_name] is subclient

        # Import and verify type
        module_path, class_name = expected_class_path.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        expected_class = getattr(module, class_name)
        assert isinstance(subclient, expected_class)


@pytest.mark.unit
class TestSubclientCaching:
    """Unit tests for subclient caching behavior."""

    def test_subclient_caching_same_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple accesses should return same cached instance."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient()

        # Access multiple times
        datasets1 = client.datasets
        datasets2 = client.datasets
        datasets3 = client.datasets

        # All should be the exact same instance
        assert datasets1 is datasets2
        assert datasets2 is datasets3

    def test_different_subclients_independent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Different subclients should be independent instances."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient()

        # Access all subclients
        datasets = client.datasets
        experiments = client.experiments
        ml = client.ml
        projects = client.projects
        spans = client.spans
        annotation_configs = client.annotation_configs

        # All should be different instances
        assert datasets is not experiments
        assert experiments is not ml
        assert ml is not projects
        assert projects is not spans
        assert spans is not annotation_configs

        # All should be in cache
        assert len(client._lazy_cache) == 6

    def test_repr_shows_loaded_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repr should show which subclients are loaded vs lazy."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient()
        repr_before = repr(client)

        # All should show as lazy
        assert "'datasets': lazy" in repr_before
        assert "'experiments': lazy" in repr_before

        # Load one
        _ = client.datasets
        repr_after = repr(client)

        # datasets should show as loaded, others lazy
        assert "'datasets': loaded" in repr_after
        assert "'experiments': lazy" in repr_after


@pytest.mark.unit
class TestThreadSafetyRealClients:
    """Unit tests for thread safety with real clients."""

    def test_concurrent_access_same_subclient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent access to same subclient should be thread-safe."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient()
        results: list = []

        def access_datasets() -> None:
            results.append(client.datasets)

        # Launch 10 threads concurrently
        threads = [threading.Thread(target=access_datasets) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should get the same instance
        assert len(results) == 10
        assert all(r is results[0] for r in results)

    def test_concurrent_access_different_subclients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent access to different subclients should work."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient()
        results: dict = {}

        def access_datasets() -> None:
            results["datasets"] = client.datasets

        def access_experiments() -> None:
            results["experiments"] = client.experiments

        def access_ml() -> None:
            results["ml"] = client.ml

        threads = [
            threading.Thread(target=access_datasets),
            threading.Thread(target=access_experiments),
            threading.Thread(target=access_ml),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All three should be loaded
        assert len(results) == 3
        assert results["datasets"] is not None
        assert results["experiments"] is not None
        assert results["ml"] is not None


@pytest.mark.unit
class TestClearCacheRealFilesystem:
    """Unit tests for cache clearing with real filesystem."""

    def test_clear_cache_deletes_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """clear_cache should delete real cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "file1.txt").write_text("test1")
        (cache_dir / "file2.txt").write_text("test2")

        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient(arize_directory=str(tmp_path))

        # Verify directory exists
        assert cache_dir.exists()

        # Clear cache
        client.clear_cache()

        # Directory should be deleted
        assert not cache_dir.exists()

    def test_clear_cache_with_nested_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """clear_cache should recursively delete nested directories."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        subdir = cache_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested_file.txt").write_text("nested")

        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient(arize_directory=str(tmp_path))
        client.clear_cache()

        # Everything should be deleted
        assert not cache_dir.exists()
        assert not subdir.exists()

    def test_clear_cache_nonexistent_raises_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """clear_cache with nonexistent directory should handle gracefully."""
        nonexistent = tmp_path / "nonexistent"

        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient(arize_directory=str(nonexistent))

        # Should raise NotADirectoryError (current behavior based on code)
        with pytest.raises(NotADirectoryError):
            client.clear_cache()


@pytest.mark.unit
class TestClientWithSubclients:
    """Unit tests for client with real subclient usage."""

    def test_subclients_have_sdk_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Subclients should receive SDK configuration."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient()
        datasets = client.datasets

        # Subclient should have _sdk_config (private attribute)
        assert hasattr(datasets, "_sdk_config")
        assert datasets._sdk_config is client.sdk_config

    def test_subclients_have_generated_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Subclients should receive generated client when needed."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient()
        datasets = client.datasets

        # Check if subclient has generated_client (if it requires it)
        if hasattr(datasets, "_generated_client"):
            assert datasets._generated_client is not None

    def test_all_subclients_loadable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All subclients should be loadable without errors."""
        monkeypatch.setenv("ARIZE_API_KEY", "test_key")

        client = ArizeClient()

        # Should all load successfully
        assert client.datasets is not None
        assert client.experiments is not None
        assert client.ml is not None
        assert client.projects is not None
        assert client.spans is not None
        assert client.annotation_configs is not None

        # All should be in cache
        assert len(client._lazy_cache) == 6
