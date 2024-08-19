import sys

import pytest

if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8 or higher", allow_module_level=True)

from typing import Any

import pandas as pd
from arize.experimental.datasets.experiments.evaluators.base import EvaluationResult, Evaluator
from arize.experimental.datasets.experiments.functions import run_experiment


# Define a simple evaluator
class MyEval(Evaluator):
    def evaluate(self, *, output, example, **_: Any) -> EvaluationResult:
        return EvaluationResult(
            explanation="eval explanation",
            score=1,
            label="good",
            metadata={"output": output, "input": example.dataset_row["question"]},
        )


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
    def task(example):
        question = example.dataset_row["question"]
        return f"Answer to {question}"

    exp_df = run_experiment(
        dataset=dataset,
        task=task,
        evaluators=[MyEval()],
        experiment_name="test_experiment",
    )
    # output df should have 2 rows x 8 cols
    assert exp_df.shape == (2, 8)
    # expected col names
    assert exp_df.columns.tolist() == [
        "id",
        "example_id",
        "result",
        "eval.MyEval.score",
        "eval.MyEval.label",
        "eval.MyEval.explanation",
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
