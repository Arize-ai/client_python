"""Unit tests for src/arize/experiments/functions.py."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from arize.experiments.evaluators.types import EvaluationResultFieldNames
from arize.experiments.functions import (
    run_experiment,
    transform_to_experiment_format,
)
from arize.experiments.types import ExperimentTaskFieldNames

METADATA = SpanAttributes.METADATA


def test_transform_renames_eval_columns_into_target_format() -> None:
    df = pd.DataFrame(
        {
            "example_id": ["ex1", "ex2"],
            "output": ["r1", "r2"],
            "score_col": [0.9, 0.8],
            "label_col": ["good", "ok"],
            "explanation_col": ["e1", "e2"],
        }
    )

    result = transform_to_experiment_format(
        experiment_runs=df,
        task_fields=ExperimentTaskFieldNames(
            example_id="example_id", output="output"
        ),
        evaluator_fields={
            "correctness": EvaluationResultFieldNames(
                score="score_col",
                label="label_col",
                explanation="explanation_col",
            )
        },
    )

    assert list(result["eval.correctness.score"]) == [0.9, 0.8]
    assert list(result["eval.correctness.label"]) == ["good", "ok"]
    assert list(result["eval.correctness.explanation"]) == ["e1", "e2"]
    assert "score_col" not in result.columns
    assert "label_col" not in result.columns
    assert "explanation_col" not in result.columns


def test_transform_preserves_eval_columns_when_already_in_target_format() -> (
    None
):
    """Regression test for issue #70648.

    When the input column names already match the SDK's target output format,
    data was being silently dropped because the column was assigned and then
    immediately dropped.
    """
    df = pd.DataFrame(
        {
            "example_id": ["ex1", "ex2", "ex3"],
            "output": ["r1", "r2", "r3"],
            "eval.correctness.score": [0.9, 0.8, 0.95],
            "eval.correctness.label": ["good", "ok", "good"],
            "eval.correctness.explanation": ["exp1", "exp2", "exp3"],
        }
    )

    result = transform_to_experiment_format(
        experiment_runs=df,
        task_fields=ExperimentTaskFieldNames(
            example_id="example_id", output="output"
        ),
        evaluator_fields={
            "correctness": EvaluationResultFieldNames(
                score="eval.correctness.score",
                label="eval.correctness.label",
                explanation="eval.correctness.explanation",
            )
        },
    )

    assert list(result["eval.correctness.score"]) == [0.9, 0.8, 0.95]
    assert list(result["eval.correctness.label"]) == ["good", "ok", "good"]
    assert list(result["eval.correctness.explanation"]) == [
        "exp1",
        "exp2",
        "exp3",
    ]


def test_transform_renames_metadata_columns() -> None:
    df = pd.DataFrame(
        {
            "example_id": ["ex1", "ex2"],
            "output": ["r1", "r2"],
            "score_col": [0.9, 0.8],
            "raw_version": ["v1", "v2"],
        }
    )

    result = transform_to_experiment_format(
        experiment_runs=df,
        task_fields=ExperimentTaskFieldNames(
            example_id="example_id", output="output"
        ),
        evaluator_fields={
            "correctness": EvaluationResultFieldNames(
                score="score_col",
                metadata={"version": "raw_version"},
            )
        },
    )

    assert list(result["eval.correctness.metadata.version"]) == ["v1", "v2"]
    assert "raw_version" not in result.columns


def test_transform_preserves_metadata_when_already_in_target_format() -> None:
    """Regression test for issue #70648 — metadata fields."""
    df = pd.DataFrame(
        {
            "example_id": ["ex1", "ex2"],
            "output": ["r1", "r2"],
            "eval.correctness.score": [0.9, 0.8],
            "eval.correctness.metadata.version": ["v1", "v2"],
        }
    )

    result = transform_to_experiment_format(
        experiment_runs=df,
        task_fields=ExperimentTaskFieldNames(
            example_id="example_id", output="output"
        ),
        evaluator_fields={
            "correctness": EvaluationResultFieldNames(
                score="eval.correctness.score",
                metadata={
                    "version": "eval.correctness.metadata.version",
                },
            )
        },
    )

    assert list(result["eval.correctness.score"]) == [0.9, 0.8]
    assert list(result["eval.correctness.metadata.version"]) == ["v1", "v2"]


def test_transform_only_score_in_target_format() -> None:
    """Mixed case: score collides, label/explanation use distinct source names."""
    df = pd.DataFrame(
        {
            "example_id": ["ex1", "ex2"],
            "output": ["r1", "r2"],
            "eval.correctness.score": [0.9, 0.8],
            "label_col": ["good", "ok"],
        }
    )

    result = transform_to_experiment_format(
        experiment_runs=df,
        task_fields=ExperimentTaskFieldNames(
            example_id="example_id", output="output"
        ),
        evaluator_fields={
            "correctness": EvaluationResultFieldNames(
                score="eval.correctness.score",
                label="label_col",
            )
        },
    )

    assert list(result["eval.correctness.score"]) == [0.9, 0.8]
    assert list(result["eval.correctness.label"]) == ["good", "ok"]
    assert "label_col" not in result.columns


@pytest.mark.unit
class TestRunExperimentSpanMetadata:
    """Tests that run_experiment writes the correct metadata keys to task spans."""

    def _make_tracer_and_exporter(
        self,
    ) -> tuple[TracerProvider, InMemorySpanExporter]:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return provider, exporter

    def test_span_metadata_includes_experiment_keys(self) -> None:
        provider, exporter = self._make_tracer_and_exporter()
        tracer = provider.get_tracer(__name__)
        resource = Resource.create({})

        dataset = pd.DataFrame({"id": ["ex1"], "input": ["hello"]})

        run_experiment(
            experiment_name="my-experiment",
            experiment_id="exp-abc",
            dataset=dataset,
            task=lambda row: row["input"],
            tracer=tracer,
            resource=resource,
            metadata={
                "dataset_id": "ds-123",
                "dataset_name": "my-dataset",
                "dataset_version_id": "dsv-456",
            },
        )

        spans = exporter.get_finished_spans()
        assert spans, "expected at least one span"
        # find the root task span (the one with METADATA set)
        metadata_spans = [
            s for s in spans if s.attributes and METADATA in s.attributes
        ]
        assert metadata_spans, "no span had the METADATA attribute set"
        md = json.loads(metadata_spans[0].attributes[METADATA])
        assert md["experiment_id"] == "exp-abc"
        assert md["experiment_name"] == "my-experiment"
        assert md["dataset_id"] == "ds-123"
        assert md["dataset_name"] == "my-dataset"
        assert md["dataset_version_id"] == "dsv-456"

    def test_span_metadata_omits_none_dataset_fields(self) -> None:
        """Optional dataset fields must not appear in metadata when not provided."""
        provider, exporter = self._make_tracer_and_exporter()
        tracer = provider.get_tracer(__name__)
        resource = Resource.create({})

        dataset = pd.DataFrame({"id": ["ex1"], "input": ["hello"]})

        run_experiment(
            experiment_name="my-experiment",
            experiment_id="exp-abc",
            dataset=dataset,
            task=lambda row: row["input"],
            tracer=tracer,
            resource=resource,
        )

        spans = exporter.get_finished_spans()
        metadata_spans = [
            s for s in spans if s.attributes and METADATA in s.attributes
        ]
        assert metadata_spans
        md = json.loads(metadata_spans[0].attributes[METADATA])
        assert md["experiment_id"] == "exp-abc"
        assert md["experiment_name"] == "my-experiment"
        assert "dataset_id" not in md
        assert "dataset_name" not in md
        assert "dataset_version_id" not in md

    def test_span_metadata_includes_user_fields(self) -> None:
        """User identity fields must appear in metadata when provided."""
        provider, exporter = self._make_tracer_and_exporter()
        tracer = provider.get_tracer(__name__)
        resource = Resource.create({})

        dataset = pd.DataFrame({"id": ["ex1"], "input": ["hello"]})

        run_experiment(
            experiment_name="my-experiment",
            experiment_id="exp-abc",
            dataset=dataset,
            task=lambda row: row["input"],
            tracer=tracer,
            resource=resource,
            metadata={
                "user_id": "user-123",
                "user_name": "Jane Doe",
                "user_email": "jane@example.com",
            },
        )

        spans = exporter.get_finished_spans()
        metadata_spans = [
            s for s in spans if s.attributes and METADATA in s.attributes
        ]
        assert metadata_spans
        md = json.loads(metadata_spans[0].attributes[METADATA])
        assert md["user_id"] == "user-123"
        assert md["user_name"] == "Jane Doe"
        assert md["user_email"] == "jane@example.com"

    def test_span_metadata_omits_none_user_fields(self) -> None:
        """User identity fields must not appear in metadata when not provided."""
        provider, exporter = self._make_tracer_and_exporter()
        tracer = provider.get_tracer(__name__)
        resource = Resource.create({})

        dataset = pd.DataFrame({"id": ["ex1"], "input": ["hello"]})

        run_experiment(
            experiment_name="my-experiment",
            experiment_id="exp-abc",
            dataset=dataset,
            task=lambda row: row["input"],
            tracer=tracer,
            resource=resource,
        )

        spans = exporter.get_finished_spans()
        metadata_spans = [
            s for s in spans if s.attributes and METADATA in s.attributes
        ]
        assert metadata_spans
        md = json.loads(metadata_spans[0].attributes[METADATA])
        assert "user_id" not in md
        assert "user_name" not in md
        assert "user_email" not in md
