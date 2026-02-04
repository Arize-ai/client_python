"""Unit tests for src/arize/pre_releases.py."""

import logging

import pytest

from arize.pre_releases import (
    ReleaseStage,
    _format_prerelease_message,
    prerelease_endpoint,
)
from arize.version import __version__


@pytest.mark.unit
class TestReleaseStage:
    """Tests for ReleaseStage enum."""

    @pytest.mark.parametrize(
        "stage,expected_value",
        [
            (ReleaseStage.ALPHA, "alpha"),
            (ReleaseStage.BETA, "beta"),
        ],
    )
    def test_release_stage_value(
        self, stage: ReleaseStage, expected_value: str
    ) -> None:
        """Verify enum has correct value."""
        assert stage.value == expected_value


@pytest.mark.unit
class TestFormatPrereleaseMessage:
    """Tests for _format_prerelease_message() function."""

    @pytest.mark.parametrize(
        "stage,expected_article_phrase",
        [
            (ReleaseStage.ALPHA, "an alpha"),
            (ReleaseStage.BETA, "a beta"),
        ],
    )
    def test_uses_correct_article(
        self, stage: ReleaseStage, expected_article_phrase: str
    ) -> None:
        """Stage should use correct article (a/an)."""
        msg = _format_prerelease_message(key="test", stage=stage)
        assert expected_article_phrase in msg

    def test_includes_version(self) -> None:
        """Message should include the SDK version."""
        msg = _format_prerelease_message(key="test", stage=ReleaseStage.ALPHA)
        assert __version__ in msg

    def test_includes_key(self) -> None:
        """Message should include the provided key."""
        msg = _format_prerelease_message(
            key="my_feature", stage=ReleaseStage.ALPHA
        )
        assert "my_feature" in msg

    @pytest.mark.parametrize(
        "stage,stage_name,article",
        [
            (ReleaseStage.ALPHA, "ALPHA", "an"),
            (ReleaseStage.BETA, "BETA", "a"),
        ],
    )
    def test_message_format(
        self, stage: ReleaseStage, stage_name: str, article: str
    ) -> None:
        """Verify complete message format."""
        msg = _format_prerelease_message(key="test_key", stage=stage)
        expected = (
            f"[{stage_name}] test_key is {article} {stage.value} API "
            f"in Arize SDK v{__version__} and may change without notice."
        )
        assert msg == expected


@pytest.mark.unit
class TestPrereleaseEndpoint:
    """Tests for prerelease_endpoint() decorator."""

    def test_returns_decorator_function(self) -> None:
        """Decorator factory should return a decorator function."""
        decorator = prerelease_endpoint(key="test", stage=ReleaseStage.ALPHA)
        assert callable(decorator)

    def test_decorator_wraps_function(self) -> None:
        """Decorated function should preserve original function metadata."""

        @prerelease_endpoint(key="test", stage=ReleaseStage.ALPHA)
        def my_func() -> str:
            """Original docstring."""
            return "result"

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "Original docstring."

    def test_first_call_emits_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """First call to decorated function should emit a warning."""
        # Clear any previous warnings from _WARNED set
        from arize import pre_releases

        pre_releases._WARNED.clear()

        @prerelease_endpoint(key="test_first_call", stage=ReleaseStage.ALPHA)
        def test_func() -> str:
            return "result"

        caplog.set_level(logging.WARNING)
        test_func()

        assert len(caplog.records) == 1
        assert "test_first_call" in caplog.text
        assert "ALPHA" in caplog.text

    def test_second_call_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Second call to the same decorated function should not emit a warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        @prerelease_endpoint(key="test_second_call", stage=ReleaseStage.ALPHA)
        def test_func() -> str:
            return "result"

        caplog.set_level(logging.WARNING)
        test_func()  # First call - should warn
        caplog.clear()
        test_func()  # Second call - should not warn

        assert len(caplog.records) == 0

    def test_different_keys_warn_independently(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Different keys should trigger warnings independently."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        @prerelease_endpoint(key="key_one", stage=ReleaseStage.ALPHA)
        def func_one() -> str:
            return "one"

        @prerelease_endpoint(key="key_two", stage=ReleaseStage.BETA)
        def func_two() -> str:
            return "two"

        caplog.set_level(logging.WARNING)

        func_one()
        assert "key_one" in caplog.text
        assert "ALPHA" in caplog.text

        caplog.clear()
        func_two()
        assert "key_two" in caplog.text
        assert "BETA" in caplog.text

    def test_function_return_value_preserved(self) -> None:
        """Decorated function should return the original function's return value."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        @prerelease_endpoint(key="test_return", stage=ReleaseStage.ALPHA)
        def test_func() -> str:
            return "expected_result"

        result = test_func()
        assert result == "expected_result"

    def test_function_args_passed_through(self) -> None:
        """Decorated function should receive positional arguments."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        @prerelease_endpoint(key="test_args", stage=ReleaseStage.ALPHA)
        def test_func(a: int, b: int) -> int:
            return a + b

        result = test_func(5, 3)
        assert result == 8

    def test_function_kwargs_passed_through(self) -> None:
        """Decorated function should receive keyword arguments."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        @prerelease_endpoint(key="test_kwargs", stage=ReleaseStage.ALPHA)
        def test_func(*, name: str, value: int) -> str:
            return f"{name}:{value}"

        result = test_func(name="test", value=42)
        assert result == "test:42"

    def test_alpha_stage_message_format(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify alpha stage produces correct message format in log."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        @prerelease_endpoint(key="alpha_test", stage=ReleaseStage.ALPHA)
        def test_func() -> None:
            pass

        caplog.set_level(logging.WARNING)
        test_func()

        assert "[ALPHA]" in caplog.text
        assert "an alpha API" in caplog.text
        assert f"v{__version__}" in caplog.text

    def test_beta_stage_message_format(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify beta stage produces correct message format in log."""
        from arize import pre_releases

        pre_releases._WARNED.clear()

        @prerelease_endpoint(key="beta_test", stage=ReleaseStage.BETA)
        def test_func() -> None:
            pass

        caplog.set_level(logging.WARNING)
        test_func()

        assert "[BETA]" in caplog.text
        assert "a beta API" in caplog.text
        assert f"v{__version__}" in caplog.text
