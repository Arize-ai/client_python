import functools
import inspect
import json
import traceback
from binascii import hexlify
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import pandas as pd
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span
from opentelemetry.trace import Status, StatusCode, Tracer
from typing_extensions import TypeAlias

from ....utils.logging import logger
from ...datasets.experiments.evaluators.executors import get_executor_on_sync_context
from ...datasets.experiments.evaluators.rate_limiters import RateLimiter
from ..utils.experiment_utils import jsonify
from .evaluators.base import Evaluator, Evaluators
from .evaluators.utils import create_evaluator
from .tracing import capture_spans, flatten
from .types import (
    EvaluationResult,
    EvaluatorName,
    Example,
    ExperimentEvaluationRun,
    ExperimentRun,
    ExperimentTask,
    _TaskSummary,
)

RateLimitErrors: TypeAlias = Union[Type[BaseException], Sequence[Type[BaseException]]]


def run_experiment(
    dataset: pd.DataFrame,
    task: ExperimentTask,
    evaluators: Optional[Evaluators] = None,
    *,
    experiment_name: Optional[str] = None,
    tracer: Optional[Tracer] = None,
    resource: Optional[Resource] = None,
    rate_limit_errors: Optional[RateLimitErrors] = None,
    concurrency: int = 3,
    exit_on_error: bool = False,
) -> pd.DataFrame:
    task_signature = inspect.signature(task)
    _validate_task_signature(task_signature)

    examples = _dataframe_to_examples(dataset)
    if not examples:
        raise ValueError("No examples found in the dataset.")

    evaluators_by_name = _evaluators_by_name(evaluators)
    root_span_name = f"Task: {get_func_name(task)}"
    root_span_kind = CHAIN

    logger.info("🧪 Experiment started.")

    def sync_run_experiment(example: Example) -> ExperimentRun:
        output = None
        error: Optional[BaseException] = None
        status = Status(StatusCode.OK)
        with ExitStack() as stack:
            span: Span = stack.enter_context(
                cm=tracer.start_as_current_span(name=root_span_name, context=Context())
            )
            stack.enter_context(capture_spans(resource))  # type: ignore
            try:
                bound_task_args = _bind_task_signature(task_signature, example)
                _output = task(*bound_task_args.args, **bound_task_args.kwargs)
                if isinstance(_output, Awaitable):
                    sync_error_message = (
                        "Task is async and cannot be run within an existing event loop. "
                        "Consider the following options:\n\n"
                        "1. Pass in a synchronous task callable.\n"
                        "2. Use `nest_asyncio.apply()` to allow nesting event loops."
                    )
                    raise RuntimeError(sync_error_message)
                else:
                    output = _output
            except BaseException as exc:
                if exit_on_error:
                    raise exc
                span.record_exception(exc)
                status = Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}")
                error = exc
                _print_experiment_error(exc, example_id=example.id, kind="task")

            output = jsonify(output)
            if example.input:
                span.set_attribute(INPUT_VALUE, example.input)
            else:
                span.set_attribute(
                    INPUT_VALUE,
                    json.dumps(obj=jsonify(example.dataset_row), ensure_ascii=False),
                )
            span.set_attribute(INPUT_MIME_TYPE, JSON.value)
            if output is not None:
                if isinstance(output, str):
                    span.set_attribute(OUTPUT_VALUE, output)
                else:
                    span.set_attribute(OUTPUT_VALUE, json.dumps(output, ensure_ascii=False))
                    span.set_attribute(OUTPUT_MIME_TYPE, JSON.value)
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, root_span_kind)
            span.set_status(status)

        assert isinstance(
            output, (dict, list, str, int, float, bool, type(None))
        ), "Output must be JSON serializable"

        exp_run = ExperimentRun(
            experiment_id=experiment_name,
            repetition_number=1,
            start_time=_decode_unix_nano(cast(int, span.start_time)),
            end_time=(
                _decode_unix_nano(cast(int, span.end_time)) if span.end_time else datetime.now()
            ),
            dataset_example_id=example.id,
            output=output,
            error=repr(error) if error else None,
            trace_id=_str_trace_id(span.get_span_context().trace_id),
        )
        return exp_run

    async def async_run_experiment(example: Example) -> ExperimentRun:
        output = None
        error: Optional[BaseException] = None
        status = Status(StatusCode.OK)
        with ExitStack() as stack:
            span: Span = stack.enter_context(
                cm=tracer.start_as_current_span(name=root_span_name, context=Context())
            )
            stack.enter_context(capture_spans(resource))

            try:
                bound_task_args = _bind_task_signature(task_signature, example)
                _output = task(*bound_task_args.args, **bound_task_args.kwargs)
                if isinstance(_output, Awaitable):
                    output = await _output
                else:
                    output = _output
            except BaseException as exc:
                if exit_on_error:
                    raise exc
                span.record_exception(exc)
                status = Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}")
                error = exc
                _print_experiment_error(exc, example_id=example.id, kind="task")
            output = jsonify(output)
            if example.input:
                span.set_attribute(INPUT_VALUE, example.input)
            else:
                span.set_attribute(
                    INPUT_VALUE,
                    json.dumps(obj=jsonify(example.dataset_row), ensure_ascii=False),
                )
            span.set_attribute(INPUT_MIME_TYPE, JSON.value)
            if output is not None:
                if isinstance(output, str):
                    span.set_attribute(OUTPUT_VALUE, output)
                else:
                    span.set_attribute(OUTPUT_VALUE, json.dumps(output, ensure_ascii=False))
                    span.set_attribute(OUTPUT_MIME_TYPE, JSON.value)
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, root_span_kind)
            span.set_status(status)

        assert isinstance(
            output, (dict, list, str, int, float, bool, type(None))
        ), "Output must be JSON serializable"

        exp_run = ExperimentRun(
            experiment_id=experiment_name,
            repetition_number=1,
            start_time=_decode_unix_nano(cast(int, span.start_time)),
            end_time=(
                _decode_unix_nano(cast(int, span.end_time)) if span.end_time else datetime.now()
            ),
            dataset_example_id=example.id,
            output=output,
            error=repr(error) if error else None,
            trace_id=_str_trace_id(span.get_span_context().trace_id),
        )
        return exp_run

    _errors: Tuple[Type[BaseException], ...]
    if not isinstance(rate_limit_errors, Sequence):
        _errors = (rate_limit_errors,)
    else:
        _errors = tuple(filter(None, rate_limit_errors))
    rate_limiters = [RateLimiter(rate_limit_error=rle) for rle in _errors]
    rate_limited_sync_run_experiment = functools.reduce(
        lambda fn, limiter: limiter.limit(fn), rate_limiters, sync_run_experiment
    )
    rate_limited_async_run_experiment = functools.reduce(
        lambda fn, limiter: limiter.alimit(fn), rate_limiters, async_run_experiment
    )

    executor = get_executor_on_sync_context(
        sync_fn=rate_limited_sync_run_experiment,
        async_fn=rate_limited_async_run_experiment,
        max_retries=0,
        exit_on_error=exit_on_error,
        fallback_return_value=None,
        tqdm_bar_format=get_tqdm_progress_bar_formatter("running tasks"),
        concurrency=concurrency,
    )

    runs, _execution_details = executor.run(examples)
    task_summary = _TaskSummary.from_task_runs(len(dataset), runs)

    if exit_on_error and (None in runs):
        # When exit_on_error is True, the result of a failed task execution is None
        # If any task execution failed, raise an error to exit early
        raise RuntimeError("An error occurred during execution of tasks.")

    out_df = pd.DataFrame()
    out_df["id"] = [run.id for run in runs]
    out_df["example_id"] = [run.dataset_example_id for run in runs]
    out_df["result"] = [run.output for run in runs]
    out_df["result.trace.id"] = [run.trace_id for run in runs]
    out_df["result.trace.timestamp"] = [int(run.start_time.timestamp() * 1e3) for run in runs]
    out_df.set_index("id", inplace=True, drop=False)
    logger.info(f"✅ Task runs completed.\n{task_summary}")

    if evaluators_by_name:
        eval_results = evaluate_experiment(
            examples=examples,
            experiment_results=runs,
            evaluators=evaluators,
            rate_limit_errors=rate_limit_errors,
            concurrency=concurrency,
            tracer=tracer,
            resource=resource,
            exit_on_error=exit_on_error,
        )

        if exit_on_error and (None in eval_results):
            raise RuntimeError("An error occurred during execution of evaluators.")

        # group evaluation results by name
        eval_results_by_name = {}
        for r in eval_results:
            if r is None:
                continue
            if r.name not in eval_results_by_name:
                eval_results_by_name[r.name] = []
            eval_results_by_name[r.name].append(r)

        for eval_name, eval_res in eval_results_by_name.items():
            eval_data = {
                "score": lambda x: get_result_attr(x, "score", None),
                "label": lambda x: get_result_attr(x, "label", None),
                "explanation": lambda x: get_result_attr(x, "explanation", None),
                "trace.id": lambda x: x.trace_id,
                "trace.timestamp": lambda x: int(x.start_time.timestamp() * 1e3),
            }

            for attr, getter in eval_data.items():
                out_df[f"eval.{eval_name}.{attr}"] = out_df.index.map(
                    {r.experiment_run_id: getter(r) for r in eval_res}
                )
            out_df = _add_metadata_to_output_df(out_df, eval_res, eval_name)
        logger.info("✅ All evaluators completed.")
    out_df.reset_index(drop=True, inplace=True)
    return out_df


def evaluate_experiment(
    examples: Sequence[Example],
    experiment_results: Sequence[ExperimentRun],
    evaluators: Evaluators,
    *,
    rate_limit_errors: Optional[RateLimitErrors] = None,
    concurrency: int = 3,
    tracer: Optional[Tracer] = None,
    resource: Optional[Resource] = None,
    exit_on_error: bool = False,
):
    evaluators_by_name = _evaluators_by_name(evaluators)
    if not evaluators_by_name:
        raise ValueError("Must specify at least one Evaluator")
    experiment_result_dict = {run.dataset_example_id: run for run in experiment_results}
    paired_list = [
        (example, experiment_result_dict[example.id])
        for example in examples
        if example.id in experiment_result_dict
    ]

    evaluation_input = [
        (example, run, evaluator)
        for (example, run), evaluator in product(paired_list, evaluators_by_name.values())
    ]

    root_span_kind = EVALUATOR

    def sync_eval_run(
        obj: Tuple[Example, ExperimentRun, Evaluator],
    ) -> ExperimentEvaluationRun:
        example, experiment_run, evaluator = obj
        result: Optional[EvaluationResult] = None
        error: Optional[BaseException] = None
        status = Status(StatusCode.OK)
        root_span_name = f"Evaluation: {evaluator.name}"
        with ExitStack() as stack:
            span: Span = stack.enter_context(
                tracer.start_as_current_span(name=root_span_name, context=Context())
            )
            stack.enter_context(capture_spans(resource))
            try:
                result = evaluator.evaluate(
                    output=deepcopy(experiment_run.output),
                    metadata=example.metadata,
                    dataset_row=example.dataset_row,
                    example=example,
                    input=example.input,
                )
            except BaseException as exc:
                if exit_on_error:
                    raise exc
                span.record_exception(exc)
                status = Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}")
                error = exc
                _print_experiment_error(
                    exc,
                    example_id=example.id,
                    kind="evaluator",
                )
            if result:
                span.set_attributes(dict(flatten(jsonify(result), recurse_on_sequence=True)))
            span.set_attribute(OPENINFERENCE_SPAN_KIND, root_span_kind)
            span.set_status(status)

        eval_run = ExperimentEvaluationRun(
            experiment_run_id=experiment_run.id,
            start_time=_decode_unix_nano(cast(int, span.start_time)),
            end_time=(
                _decode_unix_nano(cast(int, span.end_time)) if span.end_time else datetime.now()
            ),
            name=evaluator.name,
            annotator_kind=evaluator.kind,
            error=repr(error) if error else None,
            result=result,
            trace_id=_str_trace_id(span.get_span_context().trace_id),
        )
        return eval_run

    async def async_eval_run(
        obj: Tuple[Example, ExperimentRun, Evaluator],
    ) -> ExperimentEvaluationRun:
        example, experiment_run, evaluator = obj
        result: Optional[EvaluationResult] = None
        error: Optional[BaseException] = None
        status = Status(StatusCode.OK)
        root_span_name = f"Evaluation: {evaluator.name}"
        with ExitStack() as stack:
            span: Span = stack.enter_context(
                tracer.start_as_current_span(name=root_span_name, context=Context())
            )
            stack.enter_context(capture_spans(resource))
            try:
                result = await evaluator.async_evaluate(
                    output=deepcopy(experiment_run.output),
                    metadata=example.metadata,
                    dataset_row=example.dataset_row,
                    example=example,
                )
            except BaseException as exc:
                if exit_on_error:
                    raise exc
                span.record_exception(exc)
                status = Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}")
                error = exc
                _print_experiment_error(
                    exc,
                    example_id=example.id,
                    kind="evaluator",
                )
            if result:
                span.set_attributes(dict(flatten(jsonify(result), recurse_on_sequence=True)))
            span.set_attribute(OPENINFERENCE_SPAN_KIND, root_span_kind)
            span.set_status(status)
        eval_run = ExperimentEvaluationRun(
            experiment_run_id=experiment_run.id,
            start_time=_decode_unix_nano(cast(int, span.start_time)),
            end_time=(
                _decode_unix_nano(cast(int, span.end_time)) if span.end_time else datetime.now()
            ),
            name=evaluator.name,
            annotator_kind=evaluator.kind,
            error=repr(error) if error else None,
            result=result,
            trace_id=_str_trace_id(span.get_span_context().trace_id),
        )
        return eval_run

    _errors: Tuple[Type[BaseException], ...]
    if not isinstance(rate_limit_errors, Sequence):
        _errors = (rate_limit_errors,) if rate_limit_errors is not None else ()
    else:
        _errors = tuple(filter(None, rate_limit_errors))
    rate_limiters = [RateLimiter(rate_limit_error=rate_limit_error) for rate_limit_error in _errors]

    rate_limited_sync_evaluate_run = functools.reduce(
        lambda fn, limiter: limiter.limit(fn), rate_limiters, sync_eval_run
    )
    rate_limited_async_evaluate_run = functools.reduce(
        lambda fn, limiter: limiter.alimit(fn), rate_limiters, async_eval_run
    )

    executor = get_executor_on_sync_context(
        rate_limited_sync_evaluate_run,
        rate_limited_async_evaluate_run,
        max_retries=0,
        exit_on_error=exit_on_error,
        fallback_return_value=None,
        tqdm_bar_format=get_tqdm_progress_bar_formatter("running experiment evaluations"),
        concurrency=concurrency,
    )
    eval_runs, _execution_details = executor.run(evaluation_input)
    return eval_runs


def _add_metadata_to_output_df(
    output_df: pd.DataFrame, eval_runs: List[ExperimentEvaluationRun], evaluator_name: str
):
    for eval_run in eval_runs:
        if eval_run.result is None:
            continue
        metadata = eval_run.result.metadata
        for key, value in metadata.items():
            column_name = f"eval.{evaluator_name}.metadata.{key}"
            if column_name not in output_df.columns:
                output_df[column_name] = None
            # If the value is not a primitive type, try to convert it to a string
            if value is not None and not isinstance(value, (int, float, str, bool)):
                try:
                    value = str(value)
                except Exception:
                    raise ValueError(
                        f"Metadata value for key '{key}' in evaluator '{evaluator_name}' is not a primitive"
                        "type and cannot be converted to a string."
                    )
            output_df.loc[eval_run.experiment_run_id, column_name] = value
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
    *,
    tracer: Optional[Tracer] = None,
    resource: Optional[Resource] = None,
) -> ExperimentEvaluationRun:
    # validate evaluator is not async
    if inspect.iscoroutinefunction(obj[2].evaluate):
        raise ValueError("Evaluator function cannot be asynchronous.")
    example, experiment_run, evaluator = obj
    result: Optional[EvaluationResult] = None
    error: Optional[BaseException] = None
    status = Status(StatusCode.OK)
    root_span_kind = EVALUATOR
    root_span_name = f"Evaluation: {evaluator.name}"
    with ExitStack() as stack:
        span: Span = stack.enter_context(
            tracer.start_as_current_span(name=root_span_name, context=Context())
        )
        stack.enter_context(capture_spans(resource))
        try:
            result = evaluator.evaluate(
                input=example.input,
                output=deepcopy(experiment_run.output),
                expected=example.output,
                reference=example.output,
                metadata=example.metadata,
                dataset_row=example.dataset_row,
                example=example,
            )
        except BaseException as exc:
            span.record_exception(exc)
            status = Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}")
            raise
        else:
            if result:
                span.set_attributes(dict(flatten(jsonify(result), recurse_on_sequence=True)))
            span.set_attribute(OPENINFERENCE_SPAN_KIND, root_span_kind)
            span.set_status(status)
        eval_run = ExperimentEvaluationRun(
            experiment_run_id=experiment_run.id,
            start_time=_decode_unix_nano(cast(int, span.start_time)),
            end_time=(
                _decode_unix_nano(cast(int, span.end_time)) if span.end_time else datetime.now()
            ),
            name=evaluator.name,
            error=repr(error) if error else None,
            result=result,
            trace_id=_str_trace_id(span.get_span_context().trace_id),
        )
        return eval_run


def get_func_name(fn: Callable[..., Any]) -> str:
    """
    Makes a best-effort attempt to get the name of the function.
    """

    if isinstance(fn, functools.partial):
        return fn.func.__qualname__
    if hasattr(fn, "__qualname__") and not fn.__qualname__.endswith("<lambda>"):
        return fn.__qualname__.split(".<locals>.")[-1]
    return str(fn)


def _print_experiment_error(
    error: BaseException,
    /,
    *,
    example_id: str,
    kind: Literal["evaluator", "task"],
) -> None:
    """
    Prints an experiment error.
    """
    display_error = RuntimeError(f"{kind} failed for example id {repr(example_id)}")
    display_error.__cause__ = error
    formatted_exception = "".join(
        traceback.format_exception(type(display_error), display_error, display_error.__traceback__)
    )
    print("\033[91m" + formatted_exception + "\033[0m")  # prints in red


def _decode_unix_nano(time_unix_nano: int) -> datetime:
    return datetime.fromtimestamp(time_unix_nano / 1e9, tz=timezone.utc)


def _str_trace_id(id_: int) -> str:
    return hexlify(id_.to_bytes(16, "big")).decode()


def get_tqdm_progress_bar_formatter(title: str) -> str:
    """Returns a progress bar formatter for use with tqdm.

    Args:
        title (str): The title of the progress bar, displayed as a prefix.

    Returns:
        str: A formatter to be passed to the bar_format argument of tqdm.
    """
    return (
        title + " |{bar}| {n_fmt}/{total_fmt} ({percentage:3.1f}%) "
        "| ⏳ {elapsed}<{remaining} | {rate_fmt}{postfix}"
    )


INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND

CHAIN = OpenInferenceSpanKindValues.CHAIN.value
EVALUATOR = OpenInferenceSpanKindValues.EVALUATOR.value
JSON = OpenInferenceMimeTypeValues.JSON


def get_result_attr(r, attr, default=None):
    return getattr(r.result, attr, default) if r.result else default
