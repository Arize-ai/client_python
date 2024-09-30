import sys

import pytest

if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8 or higher", allow_module_level=True)

from typing import Any, Tuple

import opentelemetry.sdk.trace as trace_sdk
import pandas as pd
from arize.experimental.datasets import ArizeDatasetsClient
from arize.experimental.datasets.experiments.evaluators.base import EvaluationResult, Evaluator
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Tracer


# Define a simple evaluator
class MyEval(Evaluator):
    def evaluate(self, *, output, example, **_: Any) -> EvaluationResult:
        return EvaluationResult(
            explanation="eval explanation",
            score=1,
            label="good",
            metadata={"output": output, "input": example.dataset_row["question"]},
        )


def task(example):
    question = example.dataset_row["question"]
    return f"Answer to {question}"


dataset = pd.DataFrame(
    {
        "id": ["example_id_1", "example_id_2"],
        "question": [
            "What is the capital of France?",
            "What is the capital of Germany?",
        ],
    }
)


def test_run_experiment():
    c = ArizeDatasetsClient(developer_key="dummy_key", api_key="dummy_key")
    exp_id, exp_df = c.run_experiment(
        space_id="dummy_space_id",
        experiment_name="test_experiment",
        dataset_id="dummy_dataset_id",
        dataset_df=dataset,
        task=task,
        evaluators=[MyEval()],
        dry_run=True,
    )
    assert exp_id == ""
    # output df should have 2 rows x 12 cols
    assert exp_df.shape == (2, 12)
    # expected col names
    assert exp_df.columns.tolist() == [
        "id",
        "example_id",
        "result",
        "result.trace.id",
        "result.trace.timestamp",
        "eval.MyEval.score",
        "eval.MyEval.label",
        "eval.MyEval.explanation",
        "eval.MyEval.trace.id",
        "eval.MyEval.trace.timestamp",
        "eval.MyEval.metadata.output",
        "eval.MyEval.metadata.input",
    ]
    ## no empty cells
    assert exp_df.isnull().sum().sum() == 0
    # check metadata
    assert exp_df["eval.MyEval.metadata.input"].tolist() == [
        "What is the capital of France?",
        "What is the capital of Germany?",
    ]
    assert exp_df["eval.MyEval.metadata.output"].tolist() == [
        "Answer to What is the capital of France?",
        "Answer to What is the capital of Germany?",
    ]
    # trace.timestamp should be int (milliseconds timestmap)
    assert exp_df["result.trace.timestamp"].dtype == int
    assert exp_df["eval.MyEval.trace.timestamp"].dtype == int


class _NoOpProcessor(trace_sdk.SpanProcessor):
    def force_flush(self, *_: Any) -> bool:
        return True


def get_no_op_processor() -> Tuple[Tracer, Resource]:
    tracer_provider = trace_sdk.TracerProvider()
    span_processor = _NoOpProcessor()
    tracer_provider.add_span_processor(span_processor)
    resource = Resource.create({})
    return tracer_provider.get_tracer(__name__), resource
