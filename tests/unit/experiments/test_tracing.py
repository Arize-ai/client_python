"""Unit tests for src/arize/experiments/tracing.py."""

from __future__ import annotations

import pytest
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry.sdk.trace import TracerProvider

from arize.experiments.tracing import LLMSpanMetricsCollector

OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
LLM_KIND = OpenInferenceSpanKindValues.LLM.value
CHAIN_KIND = OpenInferenceSpanKindValues.CHAIN.value


@pytest.mark.unit
class TestLLMSpanMetricsCollector:
    """Tests for the trace-id-keyed LLM span metrics aggregator."""

    def _make_tracer(self) -> tuple[TracerProvider, LLMSpanMetricsCollector]:
        collector = LLMSpanMetricsCollector()
        provider = TracerProvider()
        provider.add_span_processor(collector)
        return provider, collector

    def test_collects_token_count_and_cost_from_llm_span(self) -> None:
        provider, collector = self._make_tracer()
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span("Task") as root:
            trace_id = root.get_span_context().trace_id
            with tracer.start_as_current_span("llm-call") as llm_span:
                llm_span.set_attribute(OPENINFERENCE_SPAN_KIND, LLM_KIND)
                llm_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 30)
                llm_span.set_attribute(SpanAttributes.LLM_COST_TOTAL, 0.002)

        assert collector.pop(trace_id) == (30, 0.002)

    def test_ignores_non_llm_spans(self) -> None:
        provider, collector = self._make_tracer()
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span("Task") as root:
            trace_id = root.get_span_context().trace_id
            with tracer.start_as_current_span("retriever") as span:
                span.set_attribute(OPENINFERENCE_SPAN_KIND, CHAIN_KIND)
                span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 99)

        assert collector.pop(trace_id) == (None, None)

    def test_falls_back_to_prompt_plus_completion_when_total_missing(
        self,
    ) -> None:
        provider, collector = self._make_tracer()
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span("Task") as root:
            trace_id = root.get_span_context().trace_id
            with tracer.start_as_current_span("llm-call") as llm_span:
                llm_span.set_attribute(OPENINFERENCE_SPAN_KIND, LLM_KIND)
                llm_span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 10
                )
                llm_span.set_attribute(
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 5
                )

        assert collector.pop(trace_id) == (15, None)

    def test_collects_cost_only_when_token_attributes_absent(self) -> None:
        provider, collector = self._make_tracer()
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span("Task") as root:
            trace_id = root.get_span_context().trace_id
            with tracer.start_as_current_span("llm-call") as llm_span:
                llm_span.set_attribute(OPENINFERENCE_SPAN_KIND, LLM_KIND)
                llm_span.set_attribute(SpanAttributes.LLM_COST_TOTAL, 0.002)

        assert collector.pop(trace_id) == (None, 0.002)

    def test_sums_across_multiple_llm_spans_in_same_trace(self) -> None:
        provider, collector = self._make_tracer()
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span("Task") as root:
            trace_id = root.get_span_context().trace_id
            for tokens, cost in [(10, 0.001), (20, 0.002)]:
                with tracer.start_as_current_span("llm-call") as llm_span:
                    llm_span.set_attribute(OPENINFERENCE_SPAN_KIND, LLM_KIND)
                    llm_span.set_attribute(
                        SpanAttributes.LLM_TOKEN_COUNT_TOTAL, tokens
                    )
                    llm_span.set_attribute(SpanAttributes.LLM_COST_TOTAL, cost)

        token_count, total_cost = collector.pop(trace_id)
        assert token_count == 30
        assert total_cost == pytest.approx(0.003)

    def test_pop_clears_the_trace_after_returning_it(self) -> None:
        provider, collector = self._make_tracer()
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span("Task") as root:
            trace_id = root.get_span_context().trace_id
            with tracer.start_as_current_span("llm-call") as llm_span:
                llm_span.set_attribute(OPENINFERENCE_SPAN_KIND, LLM_KIND)
                llm_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 30)

        assert collector.pop(trace_id) == (30, None)
        assert collector.pop(trace_id) == (None, None)

    def test_pop_on_unknown_trace_returns_none(self) -> None:
        _, collector = self._make_tracer()
        assert collector.pop(12345) == (None, None)

    def test_no_llm_spans_returns_none(self) -> None:
        provider, collector = self._make_tracer()
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span("Task") as root:
            trace_id = root.get_span_context().trace_id

        assert collector.pop(trace_id) == (None, None)
