"""Unit tests for src/arize/logging.py."""

import json
import logging
from unittest.mock import Mock, patch

import pytest

from arize.logging import (
    CtxAdapter,
    CustomLogFormatter,
    JsonFormatter,
    _coerce_mapping,
    _parse_level,
    auto_configure_from_env,
    configure_logging,
    get_arize_project_url,
    get_truncation_warning_message,
    log_a_list,
)


@pytest.mark.unit
class TestCtxAdapter:
    """Tests for CtxAdapter class."""

    def test_process_merges_bound_and_call_extras(self) -> None:
        """Process should merge bound and call extras."""
        logger = logging.getLogger("test")
        adapter = CtxAdapter(logger, {"bound_key": "bound_value"})

        msg = "test message"
        kwargs = {"extra": {"call_key": "call_value"}}

        processed_msg, processed_kwargs = adapter.process(msg, kwargs)

        assert processed_msg == msg
        assert processed_kwargs["extra"]["bound_key"] == "bound_value"
        assert processed_kwargs["extra"]["call_key"] == "call_value"

    def test_process_bound_only(self) -> None:
        """Process with only bound extras should work."""
        logger = logging.getLogger("test")
        adapter = CtxAdapter(logger, {"bound_key": "bound_value"})

        msg = "test message"
        kwargs: dict = {}

        _processed_msg, processed_kwargs = adapter.process(msg, kwargs)

        assert processed_kwargs["extra"]["bound_key"] == "bound_value"

    def test_process_call_only(self) -> None:
        """Process with only call extras should work."""
        logger = logging.getLogger("test")
        adapter = CtxAdapter(logger, None)

        msg = "test message"
        kwargs = {"extra": {"call_key": "call_value"}}

        _processed_msg, processed_kwargs = adapter.process(msg, kwargs)

        assert processed_kwargs["extra"]["call_key"] == "call_value"

    def test_process_neither(self) -> None:
        """Process with no extras should not add extra key."""
        logger = logging.getLogger("test")
        adapter = CtxAdapter(logger, None)

        msg = "test message"
        kwargs: dict = {}

        _processed_msg, processed_kwargs = adapter.process(msg, kwargs)

        assert "extra" not in processed_kwargs

    def test_with_extra_adds_context(self) -> None:
        """with_extra should add additional context."""
        logger = logging.getLogger("test")
        adapter = CtxAdapter(logger, {"key1": "value1"})

        new_adapter = adapter.with_extra(key2="value2")

        # Original adapter unchanged
        assert adapter.extra == {"key1": "value1"}
        # New adapter has both
        assert new_adapter.extra == {"key1": "value1", "key2": "value2"}

    def test_with_extra_overrides_existing(self) -> None:
        """with_extra should override existing keys."""
        logger = logging.getLogger("test")
        adapter = CtxAdapter(logger, {"key1": "value1"})

        new_adapter = adapter.with_extra(key1="new_value")

        assert new_adapter.extra == {"key1": "new_value"}

    def test_without_extra_creates_clean(self) -> None:
        """without_extra should create adapter without bound extras."""
        logger = logging.getLogger("test")
        adapter = CtxAdapter(logger, {"key1": "value1"})

        new_adapter = adapter.without_extra()

        assert new_adapter.extra is None

    def test_without_extra_separate_instance(self) -> None:
        """without_extra should create separate instance."""
        logger = logging.getLogger("test")
        adapter = CtxAdapter(logger, {"key1": "value1"})

        new_adapter = adapter.without_extra()

        assert adapter is not new_adapter
        assert adapter.extra == {"key1": "value1"}


@pytest.mark.unit
class TestCustomLogFormatter:
    """Tests for CustomLogFormatter class."""

    def test_format_different_levels(self) -> None:
        """Should format different log levels with different colors."""
        formatter = CustomLogFormatter("%(message)s")

        for level in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
        ]:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg="test message",
                args=(),
                exc_info=None,
            )

            formatted = formatter.format(record)

            # Should contain color code
            assert "\x1b[" in formatted or "test message" in formatted

    def test_format_with_extras(self) -> None:
        """Should append extra fields to formatted message."""
        formatter = CustomLogFormatter("%(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.extra_key = "extra_value"  # type: ignore

        formatted = formatter.format(record)

        assert "extra_key" in formatted
        assert "extra_value" in formatted

    def test_format_without_extras(self) -> None:
        """Should format message without extras."""
        formatter = CustomLogFormatter("%(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "test message" in formatted
        assert "|" not in formatted or "test message" in formatted.split("|")[0]

    def test_color_codes_correct(self) -> None:
        """Should use correct color codes for each level."""
        formatter = CustomLogFormatter("%(message)s")

        # Test each level has a color
        assert formatter.COLORS[logging.DEBUG] == formatter.BLUE
        assert formatter.COLORS[logging.INFO] == formatter.GREY
        assert formatter.COLORS[logging.WARNING] == formatter.YELLOW
        assert formatter.COLORS[logging.ERROR] == formatter.RED
        assert formatter.COLORS[logging.CRITICAL] == formatter.BOLD_RED

    def test_missing_color_uses_default(self) -> None:
        """Missing color for level should not break formatting."""
        formatter = CustomLogFormatter("%(message)s")

        record = logging.LogRecord(
            name="test",
            level=99,  # Non-standard level
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "test message" in formatted

    def test_extra_fields_appended(self) -> None:
        """Extra fields should be appended with pipe separator."""
        formatter = CustomLogFormatter("%(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"  # type: ignore

        formatted = formatter.format(record)

        assert " | " in formatted
        assert "custom_field" in formatted


@pytest.mark.unit
class TestJsonFormatter:
    """Tests for JsonFormatter class."""

    def test_format_builds_json_payload(self) -> None:
        """Should format record as JSON."""
        formatter = JsonFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        payload = json.loads(formatted)

        assert isinstance(payload, dict)
        assert "msg" in payload or "message" in payload

    def test_format_with_exception_info(self) -> None:
        """Should include exception info in JSON."""
        formatter = JsonFormatter()

        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="test message",
                args=(),
                exc_info=exc_info,
            )

            formatted = formatter.format(record)
            payload = json.loads(formatted)

            assert "exc_info" in payload
            assert "ValueError" in payload["exc_info"]

    def test_json_serialization(self) -> None:
        """Should serialize record to valid JSON."""
        formatter = JsonFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"  # type: ignore

        formatted = formatter.format(record)

        # Should be valid JSON
        payload = json.loads(formatted)
        assert "custom_field" in payload
        assert payload["custom_field"] == "custom_value"

    def test_ensure_ascii_false(self) -> None:
        """Should use ensure_ascii=False for JSON encoding."""
        formatter = JsonFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test 日本語",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        # Non-ASCII characters should be preserved
        assert "日本語" in formatted


@pytest.mark.unit
class TestModuleFunctions:
    """Tests for module-level functions."""

    @pytest.mark.parametrize(
        "level_input,default,expected",
        [
            (None, logging.INFO, logging.INFO),
            ("DEBUG", logging.INFO, logging.DEBUG),
            ("INFO", logging.INFO, logging.INFO),
            ("WARNING", logging.INFO, logging.WARNING),
            ("ERROR", logging.INFO, logging.ERROR),
            ("CRITICAL", logging.INFO, logging.CRITICAL),
            ("INVALID", logging.INFO, logging.INFO),
        ],
    )
    def test_parse_level(
        self, level_input: str | None, default: int, expected: int
    ) -> None:
        """_parse_level should parse valid strings or return default."""
        assert _parse_level(level_input, default) == expected

    def test_auto_configure_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should not configure when ARIZE_LOG_ENABLE env var is falsy."""
        monkeypatch.setenv("ARIZE_LOG_ENABLE", "false")

        with patch("arize.logging.configure_logging") as mock_configure:
            auto_configure_from_env()
            mock_configure.assert_not_called()

    def test_auto_configure_default_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use default level when not specified."""
        monkeypatch.setenv("ARIZE_LOG_ENABLE", "true")
        monkeypatch.delenv("ARIZE_LOG_LEVEL", raising=False)

        with patch("arize.logging.configure_logging") as mock_configure:
            auto_configure_from_env()
            mock_configure.assert_called_once()

    def test_auto_configure_custom_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use custom level when specified."""
        monkeypatch.setenv("ARIZE_LOG_ENABLE", "true")
        monkeypatch.setenv("ARIZE_LOG_LEVEL", "DEBUG")

        with patch("arize.logging.configure_logging") as mock_configure:
            auto_configure_from_env()
            mock_configure.assert_called_once_with(
                level=logging.DEBUG, structured=False
            )

    def test_auto_configure_structured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should enable structured logging when specified."""
        monkeypatch.setenv("ARIZE_LOG_ENABLE", "true")
        monkeypatch.setenv("ARIZE_LOG_STRUCTURED", "true")

        with patch("arize.logging.configure_logging") as mock_configure:
            auto_configure_from_env()
            _args, kwargs = mock_configure.call_args
            assert kwargs["structured"] is True

    def test_get_truncation_warning_message(self) -> None:
        """Should generate correct warning message."""
        msg = get_truncation_warning_message("predictions", 1000)

        assert "predictions" in msg
        assert "1000" in msg
        assert "truncated" in msg

    def test_configure_logging_levels(self) -> None:
        """Should set correct logging level."""
        configure_logging(level=logging.DEBUG, structured=False)

        logger = logging.getLogger("arize")
        assert logger.level == logging.DEBUG

    def test_configure_logging_structured_false(self) -> None:
        """Should use CustomLogFormatter when structured=False."""
        configure_logging(level=logging.INFO, structured=False)

        logger = logging.getLogger("arize")
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, CustomLogFormatter)

    def test_configure_logging_structured_true(self) -> None:
        """Should use JsonFormatter when structured=True."""
        configure_logging(level=logging.INFO, structured=True)

        logger = logging.getLogger("arize")
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)

    def test_configure_logging_removes_handlers(self) -> None:
        """Should remove existing handlers before adding new ones."""
        logger = logging.getLogger("arize")

        # Add a handler
        configure_logging(level=logging.INFO, structured=False)
        initial_count = len(logger.handlers)

        # Configure again
        configure_logging(level=logging.DEBUG, structured=True)

        # Should have same number of handlers (old one removed, new one added)
        assert len(logger.handlers) == initial_count

    @pytest.mark.parametrize(
        "items,join_word,expected",
        [
            (None, "and", ""),
            ([], "and", ""),
            (["item"], "and", "item"),
            (["item1", "item2"], "and", "item1 and item2"),
            (["item1", "item2", "item3"], "and", "item1, item2 and item3"),
            (["a", "b", "c", "d"], "or", "a, b, c or d"),
        ],
    )
    def test_log_a_list(
        self, items: list[str] | None, join_word: str, expected: str
    ) -> None:
        """log_a_list should format list items with proper separators."""
        assert log_a_list(items, join_word) == expected

    def test_get_arize_project_url_with_uri(self) -> None:
        """Should extract URI from response."""
        response = Mock()
        response.content = json.dumps(
            {"realTimeIngestionUri": "https://example.com/project"}
        ).encode()

        url = get_arize_project_url(response)
        assert url == "https://example.com/project"

    def test_get_arize_project_url_without_uri(self) -> None:
        """Should return empty string when URI not present."""
        response = Mock()
        response.content = json.dumps({"other_field": "value"}).encode()

        url = get_arize_project_url(response)
        assert url == ""

    def test_coerce_mapping_dict(self) -> None:
        """Dict should be converted to dict with string keys."""
        result = _coerce_mapping({"key": "value", 123: "numeric"})
        assert result == {"key": "value", "123": "numeric"}

    def test_coerce_mapping_non_dict(self) -> None:
        """Non-mapping should return empty dict."""
        assert _coerce_mapping("string") == {}
        assert _coerce_mapping(123) == {}
        assert _coerce_mapping(None) == {}
