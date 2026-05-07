"""Unit tests for src/arize/experiments/functions.py."""

from __future__ import annotations

import pandas as pd

from arize.experiments.evaluators.types import EvaluationResultFieldNames
from arize.experiments.functions import transform_to_experiment_format
from arize.experiments.types import ExperimentTaskFieldNames


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
