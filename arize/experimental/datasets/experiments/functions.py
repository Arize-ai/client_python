import inspect
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from arize.experimental.datasets.experiments.evaluators.base import Evaluator, Evaluators
from arize.experimental.datasets.experiments.evaluators.utils import create_evaluator
from arize.experimental.datasets.experiments.types import (
    EvaluationResult,
    EvaluatorName,
    Example,
    ExperimentEvaluationRun,
    ExperimentRun,
    ExperimentTask,
)
from arize.experimental.datasets.utils.experiment_utils import jsonify


def run_experiment(
    dataset: pd.DataFrame,
    task: ExperimentTask,
    evaluators: Optional[Evaluators] = None,
    *,
    experiment_name: str = "",
) -> pd.DataFrame:
    task_signature = inspect.signature(task)
    _validate_task_signature(task_signature)
    # validate task function is not async
    if inspect.iscoroutinefunction(task):
        raise ValueError("Task function cannot be asynchronous.")

    examples = _dataframe_to_examples(dataset)
    exp_runs = []

    def run_experiment(example: Example) -> ExperimentRun:
        error: Optional[BaseException] = None
        bound_task_args = _bind_task_signature(task_signature, example)
        start = datetime.now()
        _output = task(*bound_task_args.args, **bound_task_args.kwargs)
        end = datetime.now()
        assert isinstance(
            _output, (dict, list, str, int, float, bool, type(None))
        ), f"experiment run output must be JSON serializable, but got {type(_output)} instead"
        output = jsonify(_output)
        exp_run = ExperimentRun(
            experiment_id=experiment_name,
            repetition_number=1,
            start_time=start,
            end_time=end,
            dataset_example_id=example.id,
            output=output,
            error=repr(error) if error else None,
        )
        return exp_run

    print("Start task runs...")
    for example in examples:
        run_result = run_experiment(example)
        exp_runs.append(run_result)
    print("✅ Task runs completed.")
    output_df = pd.DataFrame()
    output_df["id"] = [run.id for run in exp_runs]
    output_df["example_id"] = [run.dataset_example_id for run in exp_runs]
    output_df["result"] = [run.output for run in exp_runs]

    evaluators_by_name = _evaluators_by_name(evaluators)
    if evaluators_by_name:
        print("Start evaluator runs...")
        run_dict = {run.dataset_example_id: run for run in exp_runs}
        paired_list = [
            (example, run_dict[example.id]) for example in examples if example.id in run_dict
        ]

        for evaluator_name, evaluator in evaluators_by_name.items():
            print(f"Running evaluator: {evaluator_name}")
            eval_runs = []
            for example, exp_run in paired_list:
                eval_run = sync_evaluate_run((example, exp_run, evaluator))
                eval_runs.append(eval_run)
            output_df[f"eval.{evaluator_name}.score"] = [
                eval_run.result.score for eval_run in eval_runs
            ]
            output_df[f"eval.{evaluator_name}.label"] = [
                eval_run.result.label for eval_run in eval_runs
            ]
            output_df[f"eval.{evaluator_name}.explanation"] = [
                eval_run.result.explanation for eval_run in eval_runs
            ]
            print(f"✅ Evaluator {evaluator_name} completed.")
        print("✅ All evaluators completed.")
    return output_df


def _dataframe_to_examples(dataset: pd.DataFrame) -> List[Example]:
    for column in dataset.columns:
        if pd.api.types.is_datetime64_any_dtype(dataset[column]):
            dataset[column] = dataset[column].astype(str)
    examples = []

    for _, row in dataset.iterrows():
        example = Example(dataset_row=row.to_dict())
        examples.append(example)
    return examples


def _validate_task_signature(sig: inspect.Signature) -> None:
    # Check that the function signature has a valid signature for use as a task
    # If it does not, raise an error to exit early before running an experiment
    params = sig.parameters
    valid_named_params = {"input", "expected", "reference", "metadata", "example", "dataset_row"}
    if len(params) == 0:
        raise ValueError("Task function must have at least one parameter.")
    if len(params) > 1:
        for not_found in set(params) - valid_named_params:
            param = params[not_found]
            if (
                param.kind is inspect.Parameter.VAR_KEYWORD
                or param.default is not inspect.Parameter.empty
            ):
                continue
            raise ValueError(
                (
                    f"Invalid parameter names in task function: {', '.join(not_found)}. "
                    "Parameters names for multi-argument functions must be "
                    f"any of: {', '.join(valid_named_params)}."
                )
            )


def _bind_task_signature(sig: inspect.Signature, example: Example) -> inspect.BoundArguments:
    parameter_mapping = {
        "input": example.input,
        "expected": example.output,
        "reference": example.output,  # Alias for "expected"
        "metadata": example.metadata,
        "example": example,
        "dataset_row": example.dataset_row,
    }
    params = sig.parameters
    if len(params) == 1:
        parameter_name = next(iter(params))
        if parameter_name in parameter_mapping:
            return sig.bind(parameter_mapping[parameter_name])
        else:
            return sig.bind(parameter_mapping["input"])
    return sig.bind_partial(
        **{name: parameter_mapping[name] for name in set(parameter_mapping).intersection(params)}
    )


def _evaluators_by_name(obj: Optional[Evaluators]) -> Mapping[EvaluatorName, Evaluator]:
    evaluators_by_name: Dict[EvaluatorName, Evaluator] = {}
    if obj is None:
        return evaluators_by_name
    if isinstance(obj, Mapping):
        for name, value in obj.items():
            evaluator = (
                create_evaluator(name=name)(value) if not isinstance(value, Evaluator) else value
            )
            name = evaluator.name
            if name in evaluators_by_name:
                raise ValueError(f"Two evaluators have the same name: {name}")
            evaluators_by_name[name] = evaluator
    elif isinstance(obj, Sequence):
        for value in obj:
            evaluator = create_evaluator()(value) if not isinstance(value, Evaluator) else value
            name = evaluator.name
            if name in evaluators_by_name:
                raise ValueError(f"Two evaluators have the same name: {name}")
            evaluators_by_name[name] = evaluator
    else:
        assert not isinstance(obj, Mapping) and not isinstance(obj, Sequence)
        evaluator = create_evaluator()(obj) if not isinstance(obj, Evaluator) else obj
        name = evaluator.name
        if name in evaluators_by_name:
            raise ValueError(f"Two evaluators have the same name: {name}")
        evaluators_by_name[name] = evaluator
    return evaluators_by_name


def sync_evaluate_run(
    obj: Tuple[Example, ExperimentRun, Evaluator],
) -> ExperimentEvaluationRun:
    ## validate evaluator is not async
    if inspect.iscoroutinefunction(obj[2].evaluate):
        raise ValueError("Evaluator function cannot be asynchronous.")
    example, experiment_run, evaluator = obj
    result: Optional[EvaluationResult] = None
    error: Optional[BaseException] = None
    start = datetime.now()
    result = evaluator.evaluate(
        input=example.input,
        output=deepcopy(experiment_run.output),
        expected=example.output,
        metadata=example.metadata,
        dataset_row=example.dataset_row,
        example=example,
    )
    end = datetime.now()
    eval_run = ExperimentEvaluationRun(
        experiment_run_id=experiment_run.id,
        start_time=start,
        end_time=end,
        name=evaluator.name,
        error=repr(error) if error else None,
        result=result,
    )
    return eval_run
