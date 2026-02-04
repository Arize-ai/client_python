"""Base evaluator classes for experiment evaluation."""

from __future__ import annotations

import functools
import inspect
from abc import ABC
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from arize.experiments.evaluators.types import (
    AnnotatorKind,
    EvaluationResult,
    EvaluatorKind,
    EvaluatorName,
    EvaluatorOutput,
    JSONSerializable,
)

if TYPE_CHECKING:
    from arize.experiments.types import (
        ExampleInput,
        ExampleMetadata,
        ExampleOutput,
        TaskOutput,
    )


class Evaluator(ABC):
    """A helper super class to guide the implementation of an `Evaluator` object.

    Subclasses must implement either the `evaluate` or `async_evaluate` method.
    Implementing both methods is recommended, but not required.

    This Class is intended to be subclassed, and should not be instantiated directly.
    """

    _kind: EvaluatorKind
    _name: EvaluatorName

    @functools.cached_property
    def name(self) -> EvaluatorName:
        """Return the name of this evaluator."""
        if hasattr(self, "_name"):
            return self._name
        return self.__class__.__name__

    @functools.cached_property
    def kind(self) -> EvaluatorKind:
        """Return the kind of this evaluator (CODE or LLM)."""
        if hasattr(self, "_kind"):
            return self._kind
        return AnnotatorKind.CODE.value

    def __new__(cls, *args: object, **kwargs: object) -> Evaluator:
        """Create a new evaluator instance, preventing direct instantiation of abstract class."""
        if cls is Evaluator:
            raise TypeError(
                f"{cls.__name__} is an abstract class and should not be instantiated."
            )
        return object.__new__(cls)

    def evaluate(
        self,
        *,
        dataset_row: Mapping[str, JSONSerializable] | None = None,
        input: ExampleInput = MappingProxyType({}),
        output: TaskOutput | None = None,
        experiment_output: TaskOutput | None = None,
        dataset_output: ExampleOutput = MappingProxyType({}),
        metadata: ExampleMetadata = MappingProxyType({}),
        **kwargs: object,
    ) -> EvaluationResult:
        """Evaluate the given inputs and produce an evaluation result.

        This method should be implemented by subclasses to perform the actual
        evaluation logic. It is recommended to implement both this synchronous
        method and the asynchronous `async_evaluate` method, but it is not required.

        Args:
            dataset_row (Mapping[str, JSONSerializable] | :obj:`None`): A row from the dataset.
            input (ExampleInput): The input provided for evaluation.
            output (TaskOutput | :obj:`None`): The output produced by the task.
            experiment_output (TaskOutput | :obj:`None`): The experiment output for comparison.
            dataset_output (ExampleOutput): The expected output from the dataset.
            metadata (ExampleMetadata): Metadata associated with the example.
            **kwargs (Any): Additional keyword arguments.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        # For subclassing, one should implement either this sync method or the
        # async version. Implementing both is recommended but not required.
        raise NotImplementedError

    async def async_evaluate(
        self,
        *,
        dataset_row: Mapping[str, JSONSerializable] | None = None,
        input: ExampleInput = MappingProxyType({}),
        output: TaskOutput | None = None,
        experiment_output: TaskOutput | None = None,
        dataset_output: ExampleOutput = MappingProxyType({}),
        metadata: ExampleMetadata = MappingProxyType({}),
        **kwargs: object,
    ) -> EvaluationResult:
        """Asynchronously evaluate the given inputs and produce an evaluation result.

        This method should be implemented by subclasses to perform the actual
        evaluation logic. It is recommended to implement both this asynchronous
        method and the synchronous `evaluate` method, but it is not required.

        Args:
            dataset_row (Mapping[str, JSONSerializable] | :obj:`None`): A row from the dataset.
            input (ExampleInput): The input provided for evaluation.
            output (TaskOutput | :obj:`None`): The output produced by the task.
            experiment_output (TaskOutput | :obj:`None`): The experiment output for comparison.
            dataset_output (ExampleOutput): The expected output from the dataset.
            metadata (ExampleMetadata): Metadata associated with the example.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            EvaluationResult: The result of the evaluation.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        # For subclassing, one should implement either this async method or the
        # sync version. Implementing both is recommended but not required.
        return self.evaluate(
            dataset_row=dataset_row,
            input=input,
            output=output,
            experiment_output=experiment_output,
            dataset_output=dataset_output,
            metadata=metadata,
            **kwargs,
        )

    def __init_subclass__(
        cls, is_abstract: bool = False, **kwargs: object
    ) -> None:
        """Validate subclass implementation when inherited.

        Args:
            is_abstract: Whether the subclass is abstract and should skip validation.
            **kwargs: Additional keyword arguments for parent class.
        """
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
                if not callable(evaluate):
                    raise TypeError(
                        f"`evaluate()` method should be callable, got {type(evaluate)}"
                    )
                # need to remove the first param, i.e. `self`
                _validate_sig(
                    functools.partial(evaluate, cast("Any", None)), "evaluate"
                )
                return
            if async_evaluate := super_cls.__dict__.get(
                Evaluator.async_evaluate.__name__
            ):
                if isinstance(async_evaluate, classmethod):
                    async_evaluate = async_evaluate.__func__
                if not callable(async_evaluate):
                    raise TypeError(
                        f"`async_evaluate()` method should be callable, got {type(async_evaluate)}"
                    )
                # need to remove the first param, i.e. `self`
                _validate_sig(
                    functools.partial(async_evaluate, cast("Any", None)),
                    "async_evaluate",
                )
                return
        raise ValueError(
            f"Evaluator must implement either "
            f"`def evaluate{evaluate_fn_signature}` or "
            f"`async def async_evaluate{evaluate_fn_signature}`"
        )


def _validate_sig(fn: Callable[..., object], fn_name: str) -> None:
    sig = inspect.signature(fn)
    validate_evaluator_signature(sig)
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return
    else:
        raise ValueError(
            f"`{fn_name}` should allow variadic keyword arguments `**kwargs`"
        )


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
        "dataset_row",
        "input",
        "output",
        "experiment_output",
        "dataset_output",
        "metadata",
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
                f"Invalid parameter names in evaluation function: {not_found}. "
                "Parameters names for multi-argument functions must be "
                f"any of: {', '.join(valid_named_params)}."
            )


class CodeEvaluator(Evaluator, ABC, is_abstract=True):
    """A convenience super class for defining code evaluators.

    There are functionally no differences between this class and the `Evaluator` class,
    except that this class has a default `_kind` attribute for AnnotatorKind.CODE.
    This class is intended to be subclassed, and should not be instantiated directly.
    """

    _kind = str(AnnotatorKind.CODE)

    def __new__(cls, *args: object, **kwargs: object) -> CodeEvaluator:
        """Create a new code evaluator instance, preventing direct instantiation of abstract class."""
        if cls is CodeEvaluator:
            raise TypeError(
                f"{cls.__name__} is an abstract class and should not be instantiated."
            )
        return object.__new__(cls)


class LLMEvaluator(Evaluator, ABC, is_abstract=True):
    """A convenience super class for defining LLM evaluators.

    There are functionally no differences between this class and the `Evaluator` class,
    except that this class has a default `_kind` attribute for AnnotatorKind.LLM.
    This class is intended to be subclassed, and should not be instantiated directly.
    """

    _kind = str(AnnotatorKind.LLM)

    def __new__(cls, *args: object, **kwargs: object) -> LLMEvaluator:
        """Create a new LLM evaluator instance, preventing direct instantiation of abstract class."""
        if cls is LLMEvaluator:
            raise TypeError(
                f"{cls.__name__} is an abstract class and should not be instantiated."
            )
        return object.__new__(cls)


ExperimentEvaluator = (
    Evaluator
    | Callable[..., EvaluatorOutput]
    | Callable[..., Awaitable[EvaluatorOutput]]
)


Evaluators = (
    ExperimentEvaluator
    | Sequence[ExperimentEvaluator]
    | Mapping[EvaluatorName, ExperimentEvaluator]
)
