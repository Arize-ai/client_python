import json
import sys

import pytest

if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8 or higher", allow_module_level=True)

import random
import string

import pandas as pd

import arize.experimental.datasets.experiments.functions as helper
from arize.experimental.datasets import ArizeDatasetsClient
from arize.experimental.datasets.experiments.evaluators.base import (
    EvaluationResult,
    Evaluator,
)
from arize.experimental.datasets.experiments.types import (
    EvaluationResultColumnNames,
    ExperimentTaskResultColumnNames,
)


# Define a simple evaluator
class DummyEval(Evaluator):
    def evaluate(self, output, dataset_row, **_) -> EvaluationResult:
        return EvaluationResult(
            explanation="eval explanation",
            score=1,
            label=dataset_row["id"],
            metadata={
                "output": output,
                "input": dataset_row["question"],
                "example_id": dataset_row["id"],
            },
        )

    async def async_evaluate(
        self, output, dataset_row, **_
    ) -> EvaluationResult:
        return EvaluationResult(
            explanation="eval explanation",
            score=1,
            label=dataset_row["id"],
            metadata={
                "output": output,
                "input": dataset_row["question"],
                "example_id": dataset_row["id"],
            },
        )


class DummyEval2(Evaluator):
    def evaluate(self, *, output, dataset_row, **_) -> EvaluationResult:
        return EvaluationResult(
            explanation="eval explanation",
            score=1,
            label=dataset_row["id"],
            metadata={
                "output": output,
                "input": dataset_row["question"],
                "example_id": dataset_row["id"],
            },
        )

    async def async_evaluate(
        self, *, output, dataset_row, **_
    ) -> EvaluationResult:
        return EvaluationResult(
            explanation="eval explanation",
            score=1,
            label=dataset_row["id"],
            metadata={
                "output": output,
                "input": dataset_row["question"],
                "example_id": dataset_row["id"],
            },
        )


def dummy_task(x):
    question = x["question"]
    return f"Answer to {question}"


dataset = pd.DataFrame(
    {
        "id": [f"id_{i}" for i in range(10)],
        "question": [
            "".join(random.choices(string.ascii_letters + string.digits, k=10))
            for _ in range(10)
        ],
    }
)


def test_run_experiment():
    c = ArizeDatasetsClient(api_key="dummy_key")
    exp_id, exp_df = c.run_experiment(
        space_id="dummy_space_id",
        experiment_name="test_experiment",
        dataset_id="dummy_dataset_id",
        dataset_df=dataset,
        task=dummy_task,
        evaluators=[DummyEval(), DummyEval2()],
        dry_run=True,
    )
    assert exp_id == ""
    # output df should have 10 rows x 21 cols
    assert exp_df.shape == (10, 21)
    # expected col names
    assert exp_df.columns.tolist() == [
        "id",
        "example_id",
        "result",
        "result.trace.id",
        "result.trace.timestamp",
        "eval.DummyEval.score",
        "eval.DummyEval.label",
        "eval.DummyEval.explanation",
        "eval.DummyEval.trace.id",
        "eval.DummyEval.trace.timestamp",
        "eval.DummyEval.metadata.output",
        "eval.DummyEval.metadata.input",
        "eval.DummyEval.metadata.example_id",
        "eval.DummyEval2.score",
        "eval.DummyEval2.label",
        "eval.DummyEval2.explanation",
        "eval.DummyEval2.trace.id",
        "eval.DummyEval2.trace.timestamp",
        "eval.DummyEval2.metadata.output",
        "eval.DummyEval2.metadata.input",
        "eval.DummyEval2.metadata.example_id",
    ]
    # no empty cells
    assert exp_df.isnull().sum().sum() == 0

    for _, row in exp_df.iterrows():
        assert (
            row["example_id"]
            == row["eval.DummyEval.metadata.example_id"]
            == row["eval.DummyEval2.metadata.example_id"]
            == row["eval.DummyEval.label"]
            == row["eval.DummyEval2.label"]
        )
        assert (
            row.result
            == row["eval.DummyEval.metadata.output"]
            == row["eval.DummyEval2.metadata.output"]
        )

    # # trace.timestamp should be int (milliseconds timestamp)
    assert exp_df["result.trace.timestamp"].dtype == int
    assert exp_df["eval.DummyEval.trace.timestamp"].dtype == int
    assert exp_df["eval.DummyEval2.trace.timestamp"].dtype == int


def exception_task(dataset_row):
    if dataset_row["id"] == "id_5":
        raise ValueError("task_exception")
    else:
        question = dataset_row["question"]
        return f"Answer to {question}"


class ExceptionEval(Evaluator):
    def evaluate(self, *, output, dataset_row, **_) -> EvaluationResult:
        if dataset_row["id"] == "id_5":
            raise ValueError("eval_exception")
        return EvaluationResult(
            explanation="eval explanation", score=1, label=dataset_row["id"]
        )


def test_task_exception_handling():
    c = ArizeDatasetsClient(api_key="dummy_key")
    exp_id, exp_df = c.run_experiment(
        space_id="dummy_space_id",
        experiment_name="test_experiment",
        dataset_id="dummy_dataset_id",
        dataset_df=dataset,
        task=exception_task,
        dry_run=True,
        exit_on_error=False,
    )
    assert exp_id == ""
    # experiment df is not empty
    assert not exp_df.empty
    assert exp_df.shape == (10, 5)
    # our task function deliberately fails for id_5 so the result should be null
    assert exp_df[exp_df["example_id"] == "id_5"].result.isnull().all()
    # other rows should have non-null result
    assert not exp_df[exp_df["example_id"] != "id_5"].isnull().any().any()


def test_task_exception_handling_exit_on_error():
    # output is empty dataframe since we exit on error
    c = ArizeDatasetsClient(api_key="dummy_key")
    # expecting exception to be raised
    with pytest.raises(RuntimeError):
        exp_id, exp_df = c.run_experiment(
            space_id="dummy_space_id",
            experiment_name="test_experiment",
            dataset_id="dummy_dataset_id",
            dataset_df=dataset,
            task=exception_task,
            dry_run=True,
            exit_on_error=True,
        )


def test_evaluator_exception_handling():
    c = ArizeDatasetsClient(api_key="dummy_key")
    exp_id, exp_df = c.run_experiment(
        space_id="dummy_space_id",
        experiment_name="test_experiment",
        dataset_id="dummy_dataset_id",
        dataset_df=dataset,
        task=dummy_task,
        evaluators=[ExceptionEval()],
        dry_run=True,
        exit_on_error=False,
    )
    assert exp_id == ""
    assert not exp_df.empty
    # we don't exit on error so we should have 10 rows x 10 cols with both task and eval columns
    assert exp_df.shape == (10, 10)
    assert set(exp_df.columns) == {
        "eval.ExceptionEval.score",
        "example_id",
        "result",
        "id",
        "eval.ExceptionEval.trace.timestamp",
        "eval.ExceptionEval.explanation",
        "result.trace.id",
        "eval.ExceptionEval.trace.id",
        "eval.ExceptionEval.label",
        "result.trace.timestamp",
    }


def test_evaluator_exception_handling_exit_on_error():
    c = ArizeDatasetsClient(api_key="dummy_key")
    with pytest.raises(RuntimeError):
        _, _ = c.run_experiment(
            space_id="dummy_space_id",
            experiment_name="test_experiment",
            dataset_id="dummy_dataset_id",
            dataset_df=dataset,
            task=dummy_task,
            evaluators=[ExceptionEval()],
            dry_run=True,
            exit_on_error=True,
        )


def test_functional_evaluation():
    df = pd.DataFrame(
        {
            "id": ["id_1", "id_2"],
            "question": ["I have a question", "I have another question"],
            "attributes.input.value": ["input_value", "input_value2"],
            "attributes.output.value": ["output_value", "output_value2"],
            "attributes.metadata": [
                {"meta_key": "meta_value"},
                {"meta_key2": "meta_value2"},
            ],
        }
    )

    def task_fn(x):
        question = x["question"]
        return f"Answer to {question}"

    async def eval_fn(
        input, output, experiment_output, dataset_output, metadata, dataset_row
    ):
        md = {
            "input": input,
            "output": output,
            "experiment_output": experiment_output,
            "dataset_output": dataset_output,
            "metadata": metadata,
            "dataset_row": dataset_row,
        }
        return EvaluationResult(
            explanation="eval explanation",
            score=1,
            label=dataset_row["id"],
            metadata=md,
        )

    c = ArizeDatasetsClient(api_key="dummy_key")
    exp_id, exp_df = c.run_experiment(
        space_id="dummy_space_id",
        experiment_name="test_experiment",
        dataset_id="dummy_dataset_id",
        dataset_df=df,
        task=task_fn,
        evaluators=[eval_fn],
        dry_run=True,
        exit_on_error=False,
    )
    assert exp_id == ""
    assert exp_df.shape == (2, 16)
    assert set(exp_df.columns) == {
        "id",
        "example_id",
        "result",
        "result.trace.id",
        "result.trace.timestamp",
        "eval.eval_fn.score",
        "eval.eval_fn.label",
        "eval.eval_fn.explanation",
        "eval.eval_fn.trace.id",
        "eval.eval_fn.trace.timestamp",
        "eval.eval_fn.metadata.input",
        "eval.eval_fn.metadata.output",
        "eval.eval_fn.metadata.experiment_output",
        "eval.eval_fn.metadata.dataset_output",
        "eval.eval_fn.metadata.metadata",
        "eval.eval_fn.metadata.dataset_row",
    }
    for _, row in exp_df.iterrows():
        assert (
            row["eval.eval_fn.metadata.dataset_output"]
            != row["eval.eval_fn.metadata.experiment_output"]
        )
        assert (
            row["eval.eval_fn.metadata.experiment_output"]
            == row["eval.eval_fn.metadata.output"]
        )
    assert exp_df.isnull().sum().sum() == 0


def test_converting_exp_df():
    input_df = pd.DataFrame(
        {
            "my_id": ["ex1", "ex2"],
            "output": [{"output_1": 1}, {"output_1": 2}],
            "unused_col": ["t1", "t2"],
            "unsed_col_2": [1234567890, 1234567891],
            "quality.quality_score": [0.9, 0.8],
            "quality.quality_label": ["good", "fair"],
            "quality.quality_explanation": ["exp_1", "exp_2"],
            "quality.meta.version": ["v1", "v2"],
            "quality.meta.model": ["gpt4", "gpt3.5"],
            "quality.meta.dict": [{"key": "val"}, {"key": "val"}],
        }
    )

    output_df = helper.transform_to_experiment_format(
        input_df,
        task_columns=ExperimentTaskResultColumnNames(
            example_id="my_id",
            result="output",
        ),
        evaluator_columns={
            "QualityEvaluator": EvaluationResultColumnNames(
                score="quality.quality_score",
                label="quality.quality_label",
                explanation="quality.quality_explanation",
                metadata={
                    "version": "quality.meta.version",
                    "model": "quality.meta.model",
                    "dict": "quality.meta.dict",
                },
            )
        },
    )
    expected_columns = {
        "id",
        "example_id",
        "result",
        "eval.QualityEvaluator.score",
        "eval.QualityEvaluator.label",
        "eval.QualityEvaluator.explanation",
        "eval.QualityEvaluator.metadata.version",
        "eval.QualityEvaluator.metadata.model",
        "eval.QualityEvaluator.metadata.dict",
    }

    assert output_df.shape == (2, 9)
    assert set(output_df.columns.tolist()) == expected_columns
    # task result (if in dictionary) is converted to json str
    assert type(output_df["result"][0]) is str
    for idx, val in enumerate(output_df["result"]):
        dict_val = json.loads(val)
        assert dict_val == input_df["output"][idx]

    # metadata subfield, if a dict is converted to json str
    assert type(output_df["eval.QualityEvaluator.metadata.dict"][0]) is str
