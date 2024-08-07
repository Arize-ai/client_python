import functools
import inspect
from abc import ABC
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, Union

from arize.experimental.datasets.experiments.types import (
    EvaluationResult,
    EvaluatorName,
    EvaluatorOutput,
    ExampleInput,
    ExampleMetadata,
    ExampleOutput,
    JSONSerializable,
    TaskOutput,
)
from typing_extensions import TypeAlias


class Evaluator(ABC):
    """
    A helper super class to guide the implementation of an `Evaluator` object.
    Subclasses must implement either the `evaluate` method.

    This Class is intended to be subclassed, and should not be instantiated directly.
    """

    _name: EvaluatorName

    @property
    def name(self) -> EvaluatorName:
        if hasattr(self, "_name"):
            return self._name
        return self.__class__.__name__

    def __new__(cls, *args: Any, **kwargs: Any) -> "Evaluator":
        if cls is Evaluator:
            raise TypeError(f"{cls.__name__} is an abstract class and should not be instantiated.")
        return object.__new__(cls)

    def evaluate(
        self,
        *,
        output: Optional[TaskOutput] = None,
        expected: Optional[ExampleOutput] = None,
        dataset_row: Optional[Mapping[str, JSONSerializable]] = None,
        metadata: ExampleMetadata = MappingProxyType({}),
        input: ExampleInput = MappingProxyType({}),
    ) -> EvaluationResult:
        # For subclassing, one should implement either this sync method or the
        # async version. Implementing both is recommended but not required.
        raise NotImplementedError

    def __init_subclass__(cls, is_abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if is_abstract:
            return
        evaluate_fn_signature = inspect.signature(Evaluator.evaluate)
        for super_cls in inspect.getmro(cls):
            evaluate = super_cls.__dict__.get(Evaluator.evaluate.__name__)
            if evaluate:
                assert callable(evaluate), "`evaluate()` method should be callable"
                # need to remove the first param, i.e. `self`
                _validate_sig(functools.partial(evaluate, None), "evaluate")
                return
        raise ValueError(f"Evaluator must implement" f"`def evaluate{evaluate_fn_signature}`")


def _validate_sig(fn: Callable[..., Any], fn_name: str) -> None:
    sig = inspect.signature(fn)
    validate_evaluator_signature(sig)
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return
    else:
        raise ValueError(f"`{fn_name}` should allow variadic keyword arguments `**kwargs`")


def validate_evaluator_signature(sig: inspect.Signature) -> None:
    # Check that the wrapped function has a valid signature for use as an evaluator
    # If it does not, raise an error to exit early before running evaluations
    params = sig.parameters
    valid_named_params = {
        "input",
        "output",
        "expected",
        "reference",
        "metadata",
        "dataset_row",
        "example",
    }
    if len(params) == 0:
        raise ValueError("Evaluation function must have at least one parameter.")
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
                    f"Invalid parameter names in evaluation function: {', '.join(not_found)}. "
                    "Parameters names for multi-argument functions must be "
                    f"any of: {', '.join(valid_named_params)}."
                )
            )


ExperimentEvaluator: TypeAlias = Union[
    Evaluator,
    Callable[..., EvaluatorOutput],
    Callable[..., Awaitable[EvaluatorOutput]],
]

Evaluators: TypeAlias = Union[
    ExperimentEvaluator,
    Sequence[ExperimentEvaluator],
    Mapping[EvaluatorName, ExperimentEvaluator],
]
