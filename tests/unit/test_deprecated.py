"""Unit tests for src/arize/deprecated.py."""

import logging
import warnings

import pytest

from arize.deprecated import _format_deprecation_message, deprecated
from arize.version import __version__


@pytest.mark.unit
class TestFormatDeprecationMessage:
    """Tests for _format_deprecation_message() function."""

    def test_basic_message_format(self) -> None:
        """Test basic deprecation message with just a key."""
        msg = _format_deprecation_message(key="test_feature")
        expected = f"[DEPRECATED] test_feature is deprecated in Arize SDK v{__version__}."
        assert msg == expected

    def test_message_with_reason(self) -> None:
        """Test deprecation message includes reason when provided."""
        msg = _format_deprecation_message(
            key="test_feature", reason="no longer supported"
        )
        assert "no longer supported" in msg
        assert f"v{__version__}" in msg

    def test_message_with_alternative(self) -> None:
        """Test deprecation message includes alternative when provided."""
        msg = _format_deprecation_message(
            key="test_feature", alternative="new_feature"
        )
        assert "Use new_feature instead" in msg
        assert f"v{__version__}" in msg

    def test_message_with_reason_and_alternative(self) -> None:
        """Test deprecation message with both reason and alternative."""
        msg = _format_deprecation_message(
            key="old_method",
            reason="replaced by improved implementation",
            alternative="new_method",
        )
        expected = (
            f"[DEPRECATED] old_method is deprecated in Arize SDK v{__version__} "
            "(replaced by improved implementation). Use new_method instead."
        )
        assert msg == expected

    def test_includes_version(self) -> None:
        """Message should include the SDK version."""
        msg = _format_deprecation_message(key="test")
        assert __version__ in msg

    def test_includes_key(self) -> None:
        """Message should include the provided key."""
        msg = _format_deprecation_message(key="my_deprecated_feature")
        assert "my_deprecated_feature" in msg


@pytest.mark.unit
class TestDeprecatedDecorator:
    """Tests for deprecated() decorator."""

    def test_returns_decorator_function(self) -> None:
        """Decorator factory should return a decorator function."""
        decorator = deprecated(key="test")
        assert callable(decorator)

    def test_decorator_wraps_function(self) -> None:
        """Decorated function should preserve original function metadata."""

        @deprecated(key="test")
        def my_func() -> str:
            """Original docstring."""
            return "result"

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "Original docstring."

    def test_first_call_emits_log_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """First call to decorated function should emit a log warning."""
        from arize import deprecated as dep_module

        dep_module._WARNED.clear()

        @deprecated(key="test_first_log", emit_warning=False)
        def test_func() -> str:
            return "result"

        caplog.set_level(logging.WARNING)
        test_func()

        assert len(caplog.records) == 1
        assert "test_first_log" in caplog.text
        assert "DEPRECATED" in caplog.text

    def test_first_call_emits_deprecation_warning(self) -> None:
        """First call should emit a DeprecationWarning."""
        from arize import deprecated as dep_module

        dep_module._WARNED.clear()

        @deprecated(key="test_first_warning")
        def test_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_func()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "test_first_warning" in str(w[0].message)
            assert "DEPRECATED" in str(w[0].message)

    def test_emit_warning_false_no_deprecation_warning(self) -> None:
        """When emit_warning=False, should not emit DeprecationWarning."""
        from arize import deprecated as dep_module

        dep_module._WARNED.clear()

        @deprecated(key="test_no_warning", emit_warning=False)
        def test_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_func()

            # Should not emit any warnings
            assert len(w) == 0

    def test_second_call_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Second call to the same decorated function should not emit warnings."""
        from arize import deprecated as dep_module

        dep_module._WARNED.clear()

        @deprecated(key="test_second_call")
        def test_func() -> str:
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            caplog.set_level(logging.WARNING)

            test_func()  # First call - should warn
            first_log_count = len(caplog.records)
            first_warn_count = len(w)

            caplog.clear()
            test_func()  # Second call - should not warn

            assert first_log_count == 1
            assert first_warn_count == 1
            assert len(caplog.records) == 0
            # Warning count stays the same (no new warnings)
            assert len(w) == first_warn_count

    def test_different_keys_warn_independently(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Different keys should trigger warnings independently."""
        from arize import deprecated as dep_module

        dep_module._WARNED.clear()

        @deprecated(key="key_one", alternative="new_one")
        def func_one() -> str:
            return "one"

        @deprecated(key="key_two", reason="obsolete")
        def func_two() -> str:
            return "two"

        caplog.set_level(logging.WARNING)

        func_one()
        assert "key_one" in caplog.text
        assert "new_one" in caplog.text

        caplog.clear()
        func_two()
        assert "key_two" in caplog.text
        assert "obsolete" in caplog.text

    def test_function_return_value_preserved(self) -> None:
        """Decorated function should return the original function's return value."""
        from arize import deprecated as dep_module

        dep_module._WARNED.clear()

        @deprecated(key="test_return")
        def test_func() -> str:
            return "expected_result"

        result = test_func()
        assert result == "expected_result"

    def test_function_args_passed_through(self) -> None:
        """Decorated function should receive positional arguments."""
        from arize import deprecated as dep_module

        dep_module._WARNED.clear()

        @deprecated(key="test_args")
        def test_func(a: int, b: int) -> int:
            return a + b

        result = test_func(5, 3)
        assert result == 8

    def test_function_kwargs_passed_through(self) -> None:
        """Decorated function should receive keyword arguments."""
        from arize import deprecated as dep_module

        dep_module._WARNED.clear()

        @deprecated(key="test_kwargs")
        def test_func(*, name: str, value: int) -> str:
            return f"{name}:{value}"

        result = test_func(name="test", value=42)
        assert result == "test:42"

    def test_message_with_reason_in_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify reason appears in log when provided."""
        from arize import deprecated as dep_module

        dep_module._WARNED.clear()

        @deprecated(key="test_reason", reason="no longer maintained")
        def test_func() -> None:
            pass

        caplog.set_level(logging.WARNING)
        test_func()

        assert "no longer maintained" in caplog.text
        assert f"v{__version__}" in caplog.text

    def test_message_with_alternative_in_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify alternative appears in log when provided."""
        from arize import deprecated as dep_module

        dep_module._WARNED.clear()

        @deprecated(key="test_alt", alternative="shiny_new_method")
        def test_func() -> None:
            pass

        caplog.set_level(logging.WARNING)
        test_func()

        assert "Use shiny_new_method instead" in caplog.text
        assert f"v{__version__}" in caplog.text

    def test_stacklevel_correct(self) -> None:
        """DeprecationWarning should point to the caller's location."""
        from arize import deprecated as dep_module

        dep_module._WARNED.clear()

        @deprecated(key="test_stacklevel")
        def test_func() -> None:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            test_func()  # This line should be in the warning

            assert len(w) == 1
            # The warning should reference this file (test location)
            assert __file__ in w[0].filename
