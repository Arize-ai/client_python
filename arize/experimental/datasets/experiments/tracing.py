from __future__ import annotations

import inspect
import json
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
from openinference.semconv import trace
from openinference.semconv.trace import DocumentAttributes, SpanAttributes
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import INVALID_TRACE_ID
from typing_extensions import assert_never
from wrapt import apply_patch, resolve_path, wrap_function_wrapper


class SpanModifier:
    """
    A class that modifies spans with the specified resource attributes.
    """

    __slots__ = ("_resource",)

    def __init__(self, resource: Resource) -> None:
        self._resource = resource

    def modify_resource(self, span: ReadableSpan) -> None:
        """
        Takes a span and merges in the resource attributes specified in the constructor.

        Args:
          span: ReadableSpan: the span to modify
        """
        if (ctx := span._context) is None or ctx.span_id == INVALID_TRACE_ID:
            return
        span._resource = span._resource.merge(self._resource)


_ACTIVE_MODIFIER: ContextVar[Optional[SpanModifier]] = ContextVar("active_modifier")


def override_span(init: Callable[..., None], span: ReadableSpan, args: Any, kwargs: Any) -> None:
    init(*args, **kwargs)
    if isinstance(span_modifier := _ACTIVE_MODIFIER.get(None), SpanModifier):
        span_modifier.modify_resource(span)


_SPAN_INIT_MONKEY_PATCH_LOCK = Lock()
_SPAN_INIT_MONKEY_PATCH_COUNT = 0
_SPAN_INIT_MODULE = ReadableSpan.__init__.__module__
_SPAN_INIT_NAME = ReadableSpan.__init__.__qualname__
_SPAN_INIT_PARENT, _SPAN_INIT_ATTR, _SPAN_INIT_ORIGINAL = resolve_path(
    _SPAN_INIT_MODULE, _SPAN_INIT_NAME
)


@contextmanager
def _monkey_patch_span_init() -> Iterator[None]:
    global _SPAN_INIT_MONKEY_PATCH_COUNT
    with _SPAN_INIT_MONKEY_PATCH_LOCK:
        _SPAN_INIT_MONKEY_PATCH_COUNT += 1
        if _SPAN_INIT_MONKEY_PATCH_COUNT == 1:
            wrap_function_wrapper(
                module=_SPAN_INIT_MODULE, name=_SPAN_INIT_NAME, wrapper=override_span
            )
    yield
    with _SPAN_INIT_MONKEY_PATCH_LOCK:
        _SPAN_INIT_MONKEY_PATCH_COUNT -= 1
        if _SPAN_INIT_MONKEY_PATCH_COUNT == 0:
            apply_patch(_SPAN_INIT_PARENT, _SPAN_INIT_ATTR, _SPAN_INIT_ORIGINAL)


@contextmanager
def capture_spans(resource: Resource) -> Iterator[SpanModifier]:
    """
    A context manager that captures spans and modifies them with the specified resources.

    Args:
      resource: Resource: The resource to merge into the spans created within the context.

    Returns:
        modifier: Iterator[SpanModifier]: The span modifier that is active within the context.
    """
    modifier = SpanModifier(resource)
    with _monkey_patch_span_init():
        token = _ACTIVE_MODIFIER.set(modifier)
        yield modifier
        _ACTIVE_MODIFIER.reset(token)


"""
Span attribute keys have a special relationship with the `.` separator. When
a span attribute is ingested from protobuf, it's in the form of a key value
pair such as `("llm.token_count.completion", 123)`. What we need to do is to split
the key by the `.` separator and turn it into part of a nested dictionary such
as {"llm": {"token_count": {"completion": 123}}}. We also need to reverse this
process, which is to flatten the nested dictionary into a list of key value
pairs. This module provides functions to do both of these operations.

Note that digit keys are treated as indices of a nested array. For example,
the digits inside `("retrieval.documents.0.document.content", 'A')` and
`("retrieval.documents.1.document.content": 'B')` turn the sub-keys following
them into a nested list of dictionaries i.e.
{`retrieval: {"documents": [{"document": {"content": "A"}}, {"document":
{"content": "B"}}]}`.
"""


DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
METADATA = SpanAttributes.METADATA
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS

# attributes interpreted as JSON strings during ingestion
JSON_STRING_ATTRIBUTES = (
    DOCUMENT_METADATA,
    LLM_PROMPT_TEMPLATE_VARIABLES,
    METADATA,
    TOOL_PARAMETERS,
)

SEMANTIC_CONVENTIONS: List[str] = sorted(
    # e.g. "input.value", "llm.token_count.total", etc.
    (
        cast(str, getattr(klass, attr))
        for name in dir(trace)
        if name.endswith("Attributes") and inspect.isclass(klass := getattr(trace, name))
        for attr in dir(klass)
        if attr.isupper()
    ),
    key=len,
    reverse=True,
)  # sorted so the longer strings go first


def flatten(
    obj: Union[Mapping[str, Any], Iterable[Any]],
    *,
    prefix: str = "",
    separator: str = ".",
    recurse_on_sequence: bool = False,
    json_string_attributes: Optional[Sequence[str]] = None,
) -> Iterator[Tuple[str, Any]]:
    """
    Flatten a nested dictionary or a sequence of dictionaries into a list of
    key value pairs. If `recurse_on_sequence` is True, then the function will
    also recursively flatten nested sequences of dictionaries. If
    `json_string_attributes` is provided, then the function will interpret the
    attributes in the list as JSON strings and convert them into dictionaries.
    The `prefix` argument is used to prefix the keys in the output list, but
    it's mostly used internally to facilitate recursion.
    """
    if isinstance(obj, Mapping):
        yield from _flatten_mapping(
            obj,
            prefix=prefix,
            recurse_on_sequence=recurse_on_sequence,
            json_string_attributes=json_string_attributes,
            separator=separator,
        )
    elif isinstance(obj, Iterable):
        yield from _flatten_sequence(
            obj,
            prefix=prefix,
            recurse_on_sequence=recurse_on_sequence,
            json_string_attributes=json_string_attributes,
            separator=separator,
        )
    else:
        assert_never(obj)


def has_mapping(sequence: Iterable[Any]) -> bool:
    """
    Check if a sequence contains a dictionary. We don't flatten sequences that
    only contain primitive types, such as strings, integers, etc. Conversely,
    we'll only un-flatten digit sub-keys if it can be interpreted the index of
    an array of dictionaries.
    """
    for item in sequence:
        if isinstance(item, Mapping):
            return True
    return False


def _flatten_mapping(
    mapping: Mapping[str, Any],
    *,
    prefix: str = "",
    recurse_on_sequence: bool = False,
    json_string_attributes: Optional[Sequence[str]] = None,
    separator: str = ".",
) -> Iterator[Tuple[str, Any]]:
    """
    Flatten a nested dictionary into a list of key value pairs. If `recurse_on_sequence`
    is True, then the function will also recursively flatten nested sequences of dictionaries.
    If `json_string_attributes` is provided, then the function will interpret the attributes
    in the list as JSON strings and convert them into dictionaries. The `prefix` argument is
    used to prefix the keys in the output list, but it's mostly used internally to facilitate
    recursion.
    """
    for key, value in mapping.items():
        prefixed_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, Mapping):
            if json_string_attributes and prefixed_key.endswith(JSON_STRING_ATTRIBUTES):
                yield prefixed_key, json.dumps(value)
            else:
                yield from _flatten_mapping(
                    value,
                    prefix=prefixed_key,
                    recurse_on_sequence=recurse_on_sequence,
                    json_string_attributes=json_string_attributes,
                    separator=separator,
                )
        elif (isinstance(value, Sequence) or isinstance(value, np.ndarray)) and recurse_on_sequence:
            yield from _flatten_sequence(
                value,
                prefix=prefixed_key,
                recurse_on_sequence=recurse_on_sequence,
                json_string_attributes=json_string_attributes,
                separator=separator,
            )
        elif value is not None:
            yield prefixed_key, value


def _flatten_sequence(
    sequence: Iterable[Any],
    *,
    prefix: str = "",
    recurse_on_sequence: bool = False,
    json_string_attributes: Optional[Sequence[str]] = None,
    separator: str = ".",
) -> Iterator[Tuple[str, Any]]:
    """
    Flatten a sequence of dictionaries into a list of key value pairs. If `recurse_on_sequence`
    is True, then the function will also recursively flatten nested sequences of dictionaries.
    If `json_string_attributes` is provided, then the function will interpret the attributes
    in the list as JSON strings and convert them into dictionaries. The `prefix` argument is
    used to prefix the keys in the output list, but it's mostly used internally to facilitate
    recursion.
    """
    if isinstance(sequence, str) or not has_mapping(sequence):
        yield prefix, sequence
    for idx, obj in enumerate(sequence):
        if not isinstance(obj, Mapping):
            continue
        yield from _flatten_mapping(
            obj,
            prefix=f"{prefix}{separator}{idx}" if prefix else f"{idx}",
            recurse_on_sequence=recurse_on_sequence,
            json_string_attributes=json_string_attributes,
            separator=separator,
        )
