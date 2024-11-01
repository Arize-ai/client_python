import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional

from tqdm.auto import tqdm

from ...experiments.types import EvaluationResult, JSONSerializable
from ...utils.experiment_utils import get_func_name

if TYPE_CHECKING:
    from ..evaluators.base import Evaluator


def unwrap_json(obj: JSONSerializable) -> JSONSerializable:
    if isinstance(obj, dict) and len(obj) == 1:
        key = next(iter(obj.keys()))
        output = obj[key]
        assert isinstance(
            output, (dict, list, str, int, float, bool, type(None))
        ), "Output must be JSON serializable"
        return output
    return obj


def validate_evaluator_signature(sig: inspect.Signature) -> None:
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
    sig: inspect.Signature, **kwargs: Any
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
        else:
            return sig.bind(parameter_mapping["experiment_output"])
    return sig.bind_partial(
        **{
            name: parameter_mapping[name]
            for name in set(parameter_mapping).intersection(params)
        }
    )


def create_evaluator(
    name: Optional[str] = None,
    scorer: Optional[Callable[[Any], EvaluationResult]] = None,
) -> Callable[[Callable[..., Any]], "Evaluator"]:
    if scorer is None:
        scorer = _default_eval_scorer

    def wrapper(func: Callable[..., Any]) -> "Evaluator":
        nonlocal name
        if not name:
            name = get_func_name(func)
        assert name is not None

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
    convert_to_score: Callable[[Any], EvaluationResult],
) -> Callable[[Callable[..., Any]], "Evaluator"]:
    from ..evaluators.base import Evaluator

    def wrapper(func: Callable[..., Any]) -> "Evaluator":
        class AsyncEvaluator(Evaluator):
            def __init__(self) -> None:
                self._name = name

            @functools.wraps(func)
            async def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return await func(*args, **kwargs)

            async def async_evaluate(self, **kwargs: Any) -> EvaluationResult:
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
    convert_to_score: Callable[[Any], EvaluationResult],
) -> Callable[[Callable[..., Any]], "Evaluator"]:
    from ..evaluators.base import Evaluator

    def wrapper(func: Callable[..., Any]) -> "Evaluator":
        class SyncEvaluator(Evaluator):
            def __init__(self) -> None:
                self._name = name

            @functools.wraps(func)
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            def evaluate(self, **kwargs: Any) -> EvaluationResult:
                bound_signature = _bind_evaluator_signature(sig, **kwargs)
                result = func(*bound_signature.args, **bound_signature.kwargs)
                return convert_to_score(result)

        return SyncEvaluator()

    return wrapper


def _default_eval_scorer(result: Any) -> EvaluationResult:
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


def printif(condition: bool, *args: Any, **kwargs: Any) -> None:
    if condition:
        tqdm.write(*args, **kwargs)
