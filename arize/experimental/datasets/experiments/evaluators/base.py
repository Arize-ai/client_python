import functools
import inspect
from abc import ABC
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, Union

from typing_extensions import TypeAlias

from ...experiments.types import (
    AnnotatorKind,
    EvaluationResult,
    EvaluatorKind,
    EvaluatorName,
    EvaluatorOutput,
    ExampleInput,
    ExampleMetadata,
    ExampleOutput,
    JSONSerializable,
    TaskOutput,
)


class Evaluator(ABC):
    """
    A helper super class to guide the implementation of an `Evaluator` object.
    Subclasses must implement either the `evaluate` or `async_evaluate` method.
    Implementing both methods is recommended, but not required.

    This Class is intended to be subclassed, and should not be instantiated directly.
    """

    _kind: AnnotatorKind
    _name: EvaluatorName

    @functools.cached_property
    def name(self) -> EvaluatorName:
        if hasattr(self, "_name"):
            return self._name
        return self.__class__.__name__

    @functools.cached_property
    def kind(self) -> EvaluatorKind:
        if hasattr(self, "_kind"):
            return self._kind
        return AnnotatorKind.CODE.value

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
        **kwargs: Any,
    ) -> EvaluationResult:
        # For subclassing, one should implement either this sync method or the
        # async version. Implementing both is recommended but not required.
        raise NotImplementedError

    async def async_evaluate(
        self,
        *,
        output: Optional[TaskOutput] = None,
        expected: Optional[ExampleOutput] = None,
        dataset_row: Optional[Mapping[str, JSONSerializable]] = None,
        metadata: ExampleMetadata = MappingProxyType({}),
        input: ExampleInput = MappingProxyType({}),
        **kwargs: Any,
    ) -> EvaluationResult:
        # For subclassing, one should implement either this async method or the
        # sync version. Implementing both is recommended but not required.
        return self.evaluate(
            output=output,
            expected=expected,
            metadata=metadata,
            dataset_row=dataset_row,
            input=input,
            **kwargs,
        )

    def __init_subclass__(cls, is_abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if is_abstract:
            return
        evaluate_fn_signature = inspect.signature(Evaluator.evaluate)
        for super_cls in inspect.getmro(cls):
            if super_cls in (LLMEvaluator, Evaluator):
                break
            if evaluate := super_cls.__dict__.get(Evaluator.evaluate.__name__):
                if isinstance(evaluate, classmethod):
                    evaluate = evaluate.__func__
                assert callable(evaluate), "`evaluate()` method should be callable"
                # need to remove the first param, i.e. `self`
                _validate_sig(functools.partial(evaluate, None), "evaluate")
                return
            if async_evaluate := super_cls.__dict__.get(Evaluator.async_evaluate.__name__):
                if isinstance(async_evaluate, classmethod):
                    async_evaluate = async_evaluate.__func__
                assert callable(async_evaluate), "`async_evaluate()` method should be callable"
                # need to remove the first param, i.e. `self`
                _validate_sig(functools.partial(async_evaluate, None), "async_evaluate")
                return
        raise ValueError(
            f"Evaluator must implement either "
            f"`def evaluate{evaluate_fn_signature}` or "
            f"`async def async_evaluate{evaluate_fn_signature}`"
        )


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


class CodeEvaluator(Evaluator, ABC, is_abstract=True):
    """
    A convenience super class for defining code evaluators.

    This class is intended to be subclassed, and should not be instantiated directly.
    """

    _kind = AnnotatorKind.CODE

    def __new__(cls, *args: Any, **kwargs: Any) -> "CodeEvaluator":
        if cls is CodeEvaluator:
            raise TypeError(f"{cls.__name__} is an abstract class and should not be instantiated.")
        return object.__new__(cls)


class LLMEvaluator(Evaluator, ABC, is_abstract=True):
    """
    A convenience super class for defining LLM evaluators.

    This class is intended to be subclassed, and should not be instantiated directly.
    """

    _kind = AnnotatorKind.LLM

    def __new__(cls, *args: Any, **kwargs: Any) -> "LLMEvaluator":
        if cls is LLMEvaluator:
            raise TypeError(f"{cls.__name__} is an abstract class and should not be instantiated.")
        return object.__new__(cls)


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
