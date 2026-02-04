"""Unit tests for src/arize/_lazy.py."""

import sys
import threading
from types import ModuleType
from typing import ClassVar
from unittest.mock import Mock, patch

import pytest

from arize._lazy import (
    LazySubclientsMixin,
    OptionalDependencyError,
    _can_import,
    _dynamic_import,
    require,
)


class _TestableSubclientsMixin(LazySubclientsMixin):
    """Testable version of LazySubclientsMixin with test subclients."""

    _SUBCLIENTS: ClassVar[dict[str, tuple[str, str]]] = {
        "test_client": ("arize.test_module", "TestClient"),
        "another_client": ("arize.another_module", "AnotherClient"),
    }
    _EXTRAS: ClassVar[dict[str, str]] = {}


@pytest.mark.unit
class TestLazySubclientsMixin:
    """Tests for LazySubclientsMixin class."""

    def test_init_creates_cache_and_lock(self, mock_sdk_config: Mock) -> None:
        """Initialization should create cache dict and lock."""
        mixin = LazySubclientsMixin(mock_sdk_config)

        assert hasattr(mixin, "_lazy_cache")
        assert isinstance(mixin._lazy_cache, dict)
        assert len(mixin._lazy_cache) == 0
        assert hasattr(mixin, "_lazy_lock")
        assert isinstance(mixin._lazy_lock, type(threading.Lock()))

    def test_init_stores_sdk_config(self, mock_sdk_config: Mock) -> None:
        """Initialization should store SDK configuration."""
        mixin = LazySubclientsMixin(mock_sdk_config)

        assert mixin.sdk_config is mock_sdk_config

    def test_init_creates_gen_client_factory(
        self, mock_sdk_config: Mock
    ) -> None:
        """Initialization should create generated client factory."""
        mixin = LazySubclientsMixin(mock_sdk_config)

        assert hasattr(mixin, "_gen_client_factory")
        from arize._client_factory import GeneratedClientFactory

        assert isinstance(mixin._gen_client_factory, GeneratedClientFactory)

    def test_getattr_cache_hit_returns_immediately(
        self, mock_sdk_config: Mock
    ) -> None:
        """Cache hit should return cached instance without lazy loading."""
        mixin = _TestableSubclientsMixin(mock_sdk_config)
        cached_client = Mock()
        mixin._lazy_cache["test_client"] = cached_client

        result = mixin.__getattr__("test_client")

        assert result is cached_client

    def test_getattr_cache_miss_loads_subclient(
        self, mock_sdk_config: Mock
    ) -> None:
        """Cache miss should trigger lazy loading of subclient."""
        mixin = _TestableSubclientsMixin(mock_sdk_config)

        with patch("arize._lazy._dynamic_import") as mock_import:
            mock_module = Mock()
            mock_class = Mock()
            mock_instance = Mock()
            mock_module.TestClient = mock_class
            mock_class.return_value = mock_instance
            mock_import.return_value = mock_module

            with patch("arize._lazy.inspect.signature") as mock_sig:
                mock_sig.return_value.parameters = {}
                result = mixin.__getattr__("test_client")

                assert result is mock_instance
                assert mixin._lazy_cache["test_client"] is mock_instance
                mock_import.assert_called_once_with("arize.test_module")

    def test_getattr_passes_sdk_config_when_required(
        self, mock_sdk_config: Mock
    ) -> None:
        """Subclient requiring sdk_config should receive it."""
        mixin = _TestableSubclientsMixin(mock_sdk_config)

        with patch("arize._lazy._dynamic_import") as mock_import:
            mock_module = Mock()
            mock_class = Mock()
            mock_instance = Mock()
            mock_module.TestClient = mock_class
            mock_class.return_value = mock_instance
            mock_import.return_value = mock_module

            with patch("arize._lazy.inspect.signature") as mock_sig:
                mock_sig.return_value.parameters = {"sdk_config": Mock()}
                mixin.__getattr__("test_client")

                mock_class.assert_called_once()
                kwargs = mock_class.call_args[1]
                assert "sdk_config" in kwargs
                assert kwargs["sdk_config"] is mock_sdk_config

    def test_getattr_passes_generated_client_when_required(
        self, mock_sdk_config: Mock
    ) -> None:
        """Subclient requiring generated_client should receive it."""
        mixin = _TestableSubclientsMixin(mock_sdk_config)

        with patch("arize._lazy._dynamic_import") as mock_import:
            mock_module = Mock()
            mock_class = Mock()
            mock_instance = Mock()
            mock_module.TestClient = mock_class
            mock_class.return_value = mock_instance
            mock_import.return_value = mock_module

            with patch("arize._lazy.inspect.signature") as mock_sig:
                mock_sig.return_value.parameters = {"generated_client": Mock()}

                with patch.object(
                    mixin._gen_client_factory, "get_client"
                ) as mock_get_client:
                    mock_generated_client = Mock()
                    mock_get_client.return_value = mock_generated_client

                    mixin.__getattr__("test_client")

                    mock_class.assert_called_once()
                    kwargs = mock_class.call_args[1]
                    assert "generated_client" in kwargs
                    assert kwargs["generated_client"] is mock_generated_client

    def test_getattr_passes_both_params_when_required(
        self, mock_sdk_config: Mock
    ) -> None:
        """Subclient requiring both parameters should receive both."""
        mixin = _TestableSubclientsMixin(mock_sdk_config)

        with patch("arize._lazy._dynamic_import") as mock_import:
            mock_module = Mock()
            mock_class = Mock()
            mock_instance = Mock()
            mock_module.TestClient = mock_class
            mock_class.return_value = mock_instance
            mock_import.return_value = mock_module

            with patch("arize._lazy.inspect.signature") as mock_sig:
                mock_sig.return_value.parameters = {
                    "sdk_config": Mock(),
                    "generated_client": Mock(),
                }

                with patch.object(
                    mixin._gen_client_factory, "get_client"
                ) as mock_get_client:
                    mock_generated_client = Mock()
                    mock_get_client.return_value = mock_generated_client

                    mixin.__getattr__("test_client")

                    mock_class.assert_called_once()
                    kwargs = mock_class.call_args[1]
                    assert "sdk_config" in kwargs
                    assert "generated_client" in kwargs
                    assert kwargs["sdk_config"] is mock_sdk_config
                    assert kwargs["generated_client"] is mock_generated_client

    def test_getattr_raises_attribute_error_for_unknown(
        self, mock_sdk_config: Mock
    ) -> None:
        """Unknown attribute should raise AttributeError."""
        mixin = _TestableSubclientsMixin(mock_sdk_config)

        with pytest.raises(AttributeError, match="has no attribute 'unknown'"):
            mixin.__getattr__("unknown")

    def test_getattr_thread_safe(self, mock_sdk_config: Mock) -> None:
        """Multiple threads accessing same subclient should get same instance."""
        mixin = _TestableSubclientsMixin(mock_sdk_config)

        with patch("arize._lazy._dynamic_import") as mock_import:
            mock_module = Mock()
            mock_class = Mock()
            mock_instance = Mock()
            mock_module.TestClient = mock_class
            mock_class.return_value = mock_instance
            mock_import.return_value = mock_module

            with patch("arize._lazy.inspect.signature") as mock_sig:
                mock_sig.return_value.parameters = {}

                results = []

                def get_client() -> None:
                    results.append(mixin.__getattr__("test_client"))

                threads = [
                    threading.Thread(target=get_client) for _ in range(10)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                # All threads should get the same instance
                assert len(results) == 10
                assert all(r is results[0] for r in results)
                # Class should only be instantiated once
                assert mock_class.call_count == 1

    def test_getattr_logs_lazy_loading(
        self, mock_sdk_config: Mock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Lazy loading should log debug message."""
        mixin = _TestableSubclientsMixin(mock_sdk_config)

        with patch("arize._lazy._dynamic_import") as mock_import:
            mock_module = Mock()
            mock_class = Mock()
            mock_instance = Mock()
            mock_module.TestClient = mock_class
            mock_class.return_value = mock_instance
            mock_import.return_value = mock_module

            with patch("arize._lazy.inspect.signature") as mock_sig:
                mock_sig.return_value.parameters = {}

                with patch("arize._lazy.logger") as mock_logger:
                    mixin.__getattr__("test_client")

                    mock_logger.debug.assert_called_once()
                    call_args = mock_logger.debug.call_args[0][0]
                    assert "test_client" in call_args

    def test_dir_includes_subclients(self, mock_sdk_config: Mock) -> None:
        """dir() should include subclient names."""
        mixin = _TestableSubclientsMixin(mock_sdk_config)

        dir_result = dir(mixin)

        assert "test_client" in dir_result
        assert "another_client" in dir_result

    def test_dir_includes_parent_attributes(
        self, mock_sdk_config: Mock
    ) -> None:
        """dir() should include parent class attributes."""
        mixin = _TestableSubclientsMixin(mock_sdk_config)

        dir_result = dir(mixin)

        # Should include attributes from parent
        assert "sdk_config" in dir_result

    def test_dir_returns_sorted_list(self, mock_sdk_config: Mock) -> None:
        """dir() should return sorted list."""
        mixin = _TestableSubclientsMixin(mock_sdk_config)

        dir_result = dir(mixin)

        assert dir_result == sorted(dir_result)

    def test_multiple_subclients_independent(
        self, mock_sdk_config: Mock
    ) -> None:
        """Multiple subclients should be loaded independently."""
        mixin = _TestableSubclientsMixin(mock_sdk_config)

        with patch("arize._lazy._dynamic_import") as mock_import:

            def create_mock_module(module_path: str) -> Mock:
                mock_module = Mock()
                if "test_module" in module_path:
                    mock_class = Mock()
                    mock_class.return_value = Mock(name="TestClient")
                    mock_module.TestClient = mock_class
                else:
                    mock_class = Mock()
                    mock_class.return_value = Mock(name="AnotherClient")
                    mock_module.AnotherClient = mock_class
                return mock_module

            mock_import.side_effect = create_mock_module

            with patch("arize._lazy.inspect.signature") as mock_sig:
                mock_sig.return_value.parameters = {}

                client1 = mixin.__getattr__("test_client")
                client2 = mixin.__getattr__("another_client")

                assert client1 is not client2
                assert "test_client" in mixin._lazy_cache
                assert "another_client" in mixin._lazy_cache


@pytest.mark.unit
class TestCanImport:
    """Tests for _can_import() function."""

    @pytest.mark.parametrize(
        "module_name,expected",
        [
            ("sys", True),
            ("os", True),
            ("this_module_does_not_exist_12345", False),
        ],
    )
    def test_can_import_real_modules(
        self, module_name: str, expected: bool
    ) -> None:
        """_can_import should correctly identify importable modules."""
        assert _can_import(module_name) is expected

    @pytest.mark.parametrize(
        "error_type,error_message",
        [
            (ModuleNotFoundError, "Module not found"),
            (ImportError, "Import error"),
        ],
    )
    def test_can_import_catches_errors(
        self, error_type: type[Exception], error_message: str
    ) -> None:
        """_can_import should catch import errors and return False."""
        with patch("arize._lazy.import_module") as mock_import:
            mock_import.side_effect = error_type(error_message)
            assert _can_import("some_module") is False


@pytest.mark.unit
class TestRequire:
    """Tests for require() function."""

    def test_all_available_returns_early(self) -> None:
        """If all required modules available, should return without error."""
        require(None, ("sys", "os"))  # Should not raise

    def test_empty_required_returns_early(self) -> None:
        """If no required modules, should return without error."""
        require("some_extra", ())  # Should not raise

    def test_missing_dependency_raises_error(self) -> None:
        """Missing dependency should raise OptionalDependencyError."""
        with pytest.raises(
            OptionalDependencyError, match="Missing optional dependencies"
        ):
            require("test_extra", ("this_module_does_not_exist_12345",))

    def test_multiple_missing_dependencies(self) -> None:
        """Multiple missing dependencies should be reported."""
        with pytest.raises(OptionalDependencyError) as exc_info:
            require(
                "test_extra",
                ("nonexistent_module_1", "nonexistent_module_2"),
            )

        error_msg = str(exc_info.value)
        assert "nonexistent_module_1" in error_msg
        assert "nonexistent_module_2" in error_msg

    def test_error_message_includes_package_name(self) -> None:
        """Error message should include package name."""
        with pytest.raises(OptionalDependencyError) as exc_info:
            require(
                "test_extra", ("nonexistent_module",), pkgname="custom_package"
            )

        error_msg = str(exc_info.value)
        assert "custom_package" in error_msg

    def test_error_message_includes_extras_group(self) -> None:
        """Error message should include extras group for installation."""
        with pytest.raises(OptionalDependencyError) as exc_info:
            require("my_extra", ("nonexistent_module",))

        error_msg = str(exc_info.value)
        assert "my_extra" in error_msg
        assert "pip install" in error_msg

    def test_none_extra_key_handled(self) -> None:
        """None extra key should be handled in error message."""
        with pytest.raises(OptionalDependencyError) as exc_info:
            require(None, ("nonexistent_module",))

        error_msg = str(exc_info.value)
        assert "pip install arize[None]" in error_msg


@pytest.mark.unit
class TestDynamicImport:
    """Tests for _dynamic_import() function."""

    def test_successful_import_first_try(self) -> None:
        """Successful import on first try should return module."""
        result = _dynamic_import("sys")

        assert isinstance(result, ModuleType)
        assert result.__name__ == "sys"

    def test_successful_import_second_try(self) -> None:
        """Successful import on second try should return module."""
        with patch("arize._lazy.import_module") as mock_import:
            mock_module = Mock(spec=ModuleType)
            mock_module.__name__ = "test_module"
            # Fail first, succeed second
            mock_import.side_effect = [ModuleNotFoundError(), mock_module]

            result = _dynamic_import("test_module", retries=2)

            assert result is mock_module
            assert mock_import.call_count == 2

    def test_max_retries_exceeded(self) -> None:
        """Exceeding max retries should raise the exception."""
        with patch("arize._lazy.import_module") as mock_import:
            mock_import.side_effect = ModuleNotFoundError("Module not found")

            with pytest.raises(ModuleNotFoundError, match="Module not found"):
                _dynamic_import("test_module", retries=3)

            assert mock_import.call_count == 3

    def test_module_not_found_error_cleanup(self) -> None:
        """ModuleNotFoundError should trigger sys.modules cleanup."""
        module_name = "test_module_cleanup_mnf"

        with patch("arize._lazy.import_module") as mock_import:
            mock_import.side_effect = ModuleNotFoundError()

            # Add module to sys.modules before test
            sys.modules[module_name] = Mock()

            with pytest.raises(ModuleNotFoundError):
                _dynamic_import(module_name, retries=1)

            # Module should be removed from sys.modules
            assert module_name not in sys.modules

    def test_import_error_cleanup(self) -> None:
        """ImportError should trigger sys.modules cleanup."""
        module_name = "test_module_cleanup_ie"

        with patch("arize._lazy.import_module") as mock_import:
            mock_import.side_effect = ImportError()

            # Add module to sys.modules before test
            sys.modules[module_name] = Mock()

            with pytest.raises(ImportError):
                _dynamic_import(module_name, retries=1)

            # Module should be removed from sys.modules
            assert module_name not in sys.modules

    def test_key_error_cleanup(self) -> None:
        """KeyError should trigger sys.modules cleanup."""
        module_name = "test_module_cleanup_ke"

        with patch("arize._lazy.import_module") as mock_import:
            mock_import.side_effect = KeyError()

            # Add module to sys.modules before test
            sys.modules[module_name] = Mock()

            with pytest.raises(KeyError):
                _dynamic_import(module_name, retries=1)

            # Module should be removed from sys.modules
            assert module_name not in sys.modules

    def test_sys_modules_cleanup_verification(self) -> None:
        """Verify sys.modules cleanup behavior on error."""
        module_name = "test_module_cleanup_verify"
        # Pre-populate sys.modules
        sys.modules[module_name] = Mock()

        with patch("arize._lazy.import_module") as mock_import:
            mock_import.side_effect = ModuleNotFoundError()

            with pytest.raises(ModuleNotFoundError):
                _dynamic_import(module_name, retries=1)

            # Verify module was removed from sys.modules
            assert module_name not in sys.modules

    def test_invalid_retries_parameter(self) -> None:
        """Invalid retries parameter should raise ValueError."""
        with pytest.raises(ValueError, match="retries must be > 0"):
            _dynamic_import("sys", retries=0)

        with pytest.raises(ValueError, match="retries must be > 0"):
            _dynamic_import("sys", retries=-1)

    def test_return_value_is_module_type(self) -> None:
        """Return value should be a module type."""
        result = _dynamic_import("os", retries=1)

        assert isinstance(result, ModuleType)


@pytest.mark.unit
class TestOptionalDependencyError:
    """Tests for OptionalDependencyError exception."""

    def test_inherits_from_exception(self) -> None:
        """OptionalDependencyError should inherit from ImportError."""
        assert issubclass(OptionalDependencyError, ImportError)

    def test_can_raise_and_catch(self) -> None:
        """Should be able to raise and catch OptionalDependencyError."""
        with pytest.raises(OptionalDependencyError):
            raise OptionalDependencyError("Test error message")

        try:
            raise OptionalDependencyError("Test error")
        except OptionalDependencyError as e:
            assert "Test error" in str(e)
