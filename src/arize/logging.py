"""Logging utilities and configuration for the Arize SDK."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import requests

from arize.config import _parse_bool
from arize.constants.config import (
    DEFAULT_LOG_ENABLE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_STRUCTURED,
    ENV_LOG_ENABLE,
    ENV_LOG_LEVEL,
    ENV_LOG_STRUCTURED,
)

_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


class CtxAdapter(logging.LoggerAdapter):
    """LoggerAdapter that merges bound context with per-call extras safely."""

    def process(
        self, msg: object, kwargs: MutableMapping[str, Any]
    ) -> tuple[object, MutableMapping[str, Any]]:
        """Process the logging call by merging bound and call extras.

        Args:
            msg: The log message to process.
            kwargs: Keyword arguments from the logging call, may include 'extra' dict.

        Returns:
            tuple[object, dict[str, object]]: A tuple of (message, modified_kwargs) with
                merged extra context.
        """
        call_extra = _coerce_mapping(kwargs.pop("extra", None))
        bound_extra = _coerce_mapping(self.extra)
        merged = (
            {**bound_extra, **call_extra}
            if (bound_extra or call_extra)
            else None
        )
        if merged:
            kwargs["extra"] = merged
        return msg, kwargs

    def with_extra(self, **more: object) -> CtxAdapter:
        """Return a copy of this adapter with additional bound extras.

        Args:
            **more: Additional key-value pairs to merge into the bound extras.

        Returns:
            CtxAdapter: A new adapter instance with merged extra context.
        """
        base = _coerce_mapping(self.extra)
        base.update(_coerce_mapping(more))
        return type(self)(self.logger, base)

    def without_extra(self) -> CtxAdapter:
        """Return a copy of this adapter with *no* bound extras.

        Returns:
            CtxAdapter: A new adapter instance without any bound extra context.
        """
        return type(self)(self.logger, None)


class CustomLogFormatter(logging.Formatter):
    """Custom log formatter with color-coded output based on log level."""

    GREY = "\x1b[38;21m"
    BLUE = "\x1b[38;5;39m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[38;5;196m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: BLUE,
        logging.INFO: GREY,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def __init__(self, fmt: str) -> None:
        """Initialize the colored formatter with a format string.

        Args:
            fmt: Format string for log messages.
        """
        super().__init__(fmt=fmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with color based on log level.

        Args:
            record: The log record to format.

        Returns:
            str: Formatted and color-coded log message with any extra fields appended.
        """
        # Build the base message without any color.
        base = super().format(record)

        # Collect non-standard extras
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_RECORD_KEYS
        }

        if extras:
            # Append extras in kv form
            extras_str = " ".join(f"{k}={v!r}" for k, v in extras.items())
            base = f"{base} | {extras_str}"

        # Now color the entire line uniformly.
        color = self.COLORS.get(record.levelno, "")
        if color:
            return f"{color}{base}{self.RESET}"
        return base


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter (one JSON object per line)."""

    # fields to skip copying from record.__dict__
    _skip: ClassVar[set[str]] = set()
    # Potential fields to skip:
    # "name", "msg", "args", "levelname", "levelno", "pathname",
    # "filename", "module", "exc_info", "exc_text", "stack_info",
    # "lineno", "funcName", "created", "msecs", "relativeCreated",
    # "thread", "threadName", "processName", "process"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            str: JSON-formatted log message as a single line with all fields and extras.
        """
        payload: dict[str, object] = {
            # "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            # "logger": record.name,
            # "level": record.levelname,
            # "message": record.getMessage(),
        }

        # Include any LoggerAdapter/extra fields
        for k, v in record.__dict__.items():
            if k not in payload and k not in self._skip:
                payload[k] = v

        # Exception info, if present
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _parse_level(val: str | None, default: int = logging.INFO) -> int:
    if not val:
        return default
    return _LEVEL_MAP.get(val.strip().upper(), default)


def auto_configure_from_env() -> None:
    """If ARIZE_LOG is truthy, configure logging for 'arize' once.

    Using ARIZE_LOG_LEVEL and ARIZE_LOG_STRUCTURED if provided.
    Otherwise, do nothing (library stays quiet with NullHandler).
    """
    if not _parse_bool(os.getenv(ENV_LOG_ENABLE, DEFAULT_LOG_ENABLE)):
        return

    level = _parse_level(os.getenv(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL))
    structured = _parse_bool(
        os.getenv(ENV_LOG_STRUCTURED, DEFAULT_LOG_STRUCTURED)
    )
    configure_logging(level=level, structured=structured)


_STANDARD_RECORD_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    # asyncio
    "taskName",
    # always present in our payload already:
    "message",
    "asctime",
}


def get_truncation_warning_message(instance: str, limit: int) -> str:
    """Generate a warning message for data that exceeds character limits.

    Args:
        instance: The data instance type that may be truncated.
        limit: The character limit threshold.

    Returns:
        A formatted warning message string.
    """
    return (
        f"Attention: {instance} exceeding the {limit} character limit will be "
        "automatically truncated upon ingestion into the Arize platform. Should you require "
        "a higher limit, please reach out to our support team at support@arize.com"
    )


def configure_logging(
    level: int = logging.INFO,
    structured: bool = False,
) -> None:
    """Configure logging for the 'arize' logger.

    Args:
        level: logging level (e.g., logging.INFO, logging.DEBUG)
        to_stdout: attach a StreamHandler to stdout
        structured: if True, emit JSON logs; otherwise use color pretty logs
    """
    root = logging.getLogger("arize")
    root.setLevel(level)

    # Remove any existing handlers under 'arize'
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if structured:
        handler.setFormatter(JsonFormatter())
    else:
        fmt = "  %(name)s | %(levelname)s | %(message)s"
        handler.setFormatter(CustomLogFormatter(fmt))

    root.addHandler(handler)


def log_a_list(values: Iterable[Any] | None, join_word: str) -> str:
    """Format a list of values into a human-readable string with a joining word.

    Args:
        values: An iterable of values to format, or :obj:`None`.
        join_word: The word to use before the last item (e.g., "and", "or").

    Returns:
        A formatted string joining the values, or empty string if no values.
    """
    if values is None:
        return ""
    list_of_str = list(values)
    if len(list_of_str) == 0:
        return ""
    if len(list_of_str) == 1:
        return list_of_str[0]
    return (
        f"{', '.join(map(str, list_of_str[:-1]))} {join_word} {list_of_str[-1]}"
    )


def get_arize_project_url(response: requests.Response) -> str:
    """Extract the Arize project URL from an API response.

    Args:
        response: The HTTP response object from an Arize API call.

    Returns:
        The real-time ingestion URI if present, otherwise an empty string.
    """
    if "realTimeIngestionUri" in json.loads(response.content.decode()):
        return json.loads(response.content.decode())["realTimeIngestionUri"]
    return ""


def _coerce_mapping(obj: object) -> dict[str, object]:
    """Return a shallow dict copy if obj is a Mapping[str, Any], else {}."""
    if isinstance(obj, Mapping):
        # force keys to str to satisfy logging's expectation
        return {str(k): v for k, v in obj.items()}
    return {}
