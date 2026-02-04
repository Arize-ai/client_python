"""Utility functions for evaluator operations."""

import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from arize.experiments.evaluators.types import (
    EvaluationResult,
    JSONSerializable,
)


def get_func_name(fn: Callable[..., object]) -> str:
    """Makes a best-effort attempt to get the name of the function."""
    if isinstance(fn, functools.partial):
        return fn.func.__qualname__
    if hasattr(fn, "__qualname__") and not fn.__qualname__.endswith("<lambda>"):
        return fn.__qualname__.split(".<locals>.")[-1]
    return str(fn)


if TYPE_CHECKING:
    from ..evaluators.base import Evaluator


def unwrap_json(obj: JSONSerializable) -> JSONSerializable:
    """Unwrap a single-key JSON object to extract its value.

    Args:
        obj: A JSON-serializable object to unwrap.

    Returns:
        The unwrapped value if obj is a single-key dict, otherwise the original obj.
    """
    if isinstance(obj, dict) and len(obj) == 1:
        key = next(iter(obj.keys()))
        output = obj[key]
        if not isinstance(
            output, (dict, list, str, int, float, bool, type(None))
        ):
            raise TypeError(
                f"Evaluator output must be JSON serializable, got {type(output).__name__}"
            )
        return output
    return obj


def validate_evaluator_signature(sig: inspect.Signature) -> None:
    """Validate that a function signature is compatible for use as an evaluator.

    Args:
        sig: The function signature to validate.

    Raises:
        ValueError: If the signature is invalid for use as an evaluator.
    """
    # Check that the wrapped function has a valid signature for use as an evaluator
    # If it does not, raise an error to exit early before running evaluations
    params = sig.parameters
    valid_named_params = {
        "input",
        "output",
        "experiment_output",
        "dataset_output",
        "metadata",
        "dataset_row",
    }
    if len(params) == 0:
        raise ValueError(
            "Evaluation function must have at least one parameter."
        )
    if len(params) > 1:
        for not_found in set(params) - valid_named_params:
            param = params[not_found]
            if (
                param.kind is inspect.Parameter.VAR_KEYWORD
                or param.default is not inspect.Parameter.empty
            ):
                continue
            raise ValueError(
                f"Invalid parameter names in evaluation function: {', '.join(not_found)}. "
                "Parameters names for multi-argument functions must be "
                f"any of: {', '.join(valid_named_params)}."
            )


def _bind_evaluator_signature(
    sig: inspect.Signature, **kwargs: object
) -> inspect.BoundArguments:
    parameter_mapping = {
        "input": kwargs.get("input"),
        "output": kwargs.get("output"),
        "experiment_output": kwargs.get("experiment_output"),
        "dataset_output": kwargs.get("dataset_output"),
        "metadata": kwargs.get("metadata"),
        "dataset_row": kwargs.get("dataset_row"),
    }
    params = sig.parameters
    if len(params) == 1:
        parameter_name = next(iter(params))
        if parameter_name in parameter_mapping:
            return sig.bind(parameter_mapping[parameter_name])
        return sig.bind(parameter_mapping["experiment_output"])
    return sig.bind_partial(
        **{
            name: parameter_mapping[name]
            for name in set(parameter_mapping).intersection(params)
        }
    )


def create_evaluator(
    name: str | None = None,
    scorer: Callable[[object], EvaluationResult] | None = None,
) -> Callable[[Callable[..., object]], "Evaluator"]:
    """Create an evaluator decorator for wrapping evaluation functions.

    Args:
        name: Optional name for the evaluator. Defaults to None (uses function name).
        scorer: Optional custom scoring function. Defaults to None (uses default scorer).

    Returns:
        A decorator that wraps a function as an Evaluator instance.
    """
    if scorer is None:
        scorer = _default_eval_scorer

    def wrapper(func: Callable[..., object]) -> "Evaluator":
        nonlocal name
        if not name:
            name = get_func_name(func)
        if name is None:
            raise ValueError("Evaluator name cannot be None")

        wrapped_signature = inspect.signature(func)
        validate_evaluator_signature(wrapped_signature)

        if inspect.iscoroutinefunction(func):
            return _wrap_coroutine_evaluation_function(
                name, wrapped_signature, scorer
            )(func)

        return _wrap_sync_evaluation_function(name, wrapped_signature, scorer)(
            func
        )

    return wrapper


def _wrap_coroutine_evaluation_function(
    name: str,
    sig: inspect.Signature,
    convert_to_score: Callable[[object], EvaluationResult],
) -> Callable[[Callable[..., Awaitable[object]]], "Evaluator"]:
    from ..evaluators.base import Evaluator

    def wrapper(func: Callable[..., Awaitable[object]]) -> "Evaluator":
        class AsyncEvaluator(Evaluator):
            def __init__(self) -> None:
                self._name = name

            @functools.wraps(func)
            async def __call__(self, *args: object, **kwargs: object) -> object:
                return await func(*args, **kwargs)

            async def async_evaluate(
                self, **kwargs: object
            ) -> EvaluationResult:
                bound_signature = _bind_evaluator_signature(sig, **kwargs)
                result = await func(
                    *bound_signature.args, **bound_signature.kwargs
                )
                return convert_to_score(result)

        return AsyncEvaluator()

    return wrapper


def _wrap_sync_evaluation_function(
    name: str,
    sig: inspect.Signature,
    convert_to_score: Callable[[object], EvaluationResult],
) -> Callable[[Callable[..., object]], "Evaluator"]:
    from ..evaluators.base import Evaluator

    def wrapper(func: Callable[..., object]) -> "Evaluator":
        class SyncEvaluator(Evaluator):
            def __init__(self) -> None:
                self._name = name

            @functools.wraps(func)
            def __call__(self, *args: object, **kwargs: object) -> object:
                return func(*args, **kwargs)

            def evaluate(self, **kwargs: object) -> EvaluationResult:
                bound_signature = _bind_evaluator_signature(sig, **kwargs)
                result = func(*bound_signature.args, **bound_signature.kwargs)
                return convert_to_score(result)

        return SyncEvaluator()

    return wrapper


def _default_eval_scorer(result: object) -> EvaluationResult:
    if isinstance(result, EvaluationResult):
        return result
    if isinstance(result, bool):
        return EvaluationResult(score=float(result), label=str(result))
    if hasattr(result, "__float__"):
        return EvaluationResult(score=float(result))
    if isinstance(result, str):
        return EvaluationResult(label=result)
    if isinstance(result, (tuple, list)) and len(result) == 2:
        # If the result is a 2-tuple, the first item will be recorded as the score
        # and the second item will recorded as the explanation.
        return EvaluationResult(
            score=float(result[0]), explanation=str(result[1])
        )
    raise ValueError(f"Unsupported evaluation result type: {type(result)}")


def printif(condition: bool, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    """Print to tqdm output if the condition is true.

    Note: *args/**kwargs use Any for proper pass-through to tqdm.write().

    Args:
        condition: Whether to print the message.
        *args: Positional arguments to pass to tqdm.write.
        **kwargs: Keyword arguments to pass to tqdm.write.
    """
    if condition:
        tqdm.write(*args, **kwargs)
