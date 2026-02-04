"""Type definitions and data models for experiments."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import copy, deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from random import getrandbits
from typing import (
    NoReturn,
    cast,
)

import pandas as pd
from wrapt import ObjectProxy

from arize.experiments.evaluators.types import (
    EvaluationResult,
    JSONSerializable,
)

ExperimentId = str
ExampleId = str
RepetitionNumber = int
ExperimentRunId = str
TraceId = str


@dataclass(frozen=True)
class Example:
    """Represents an example in an experiment dataset.

    Args:
        id: The unique identifier for the example.
        updated_at: The timestamp when the example was last updated.
        input: The input data for the example.
        output: The output data for the example.
        metadata: Additional metadata for the example.
        dataset_row: The original dataset row containing the example data.
    """

    id: ExampleId = field(default_factory=str)
    updated_at: datetime = field(default_factory=datetime.now)
    input: Mapping[str, JSONSerializable] = field(default_factory=dict)
    output: Mapping[str, JSONSerializable] = field(default_factory=dict)
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)
    dataset_row: Mapping[str, JSONSerializable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize example fields from dataset_row if provided."""
        if self.dataset_row is not None:
            object.__setattr__(
                self, "dataset_row", _make_read_only(self.dataset_row)
            )
            if "attributes.input.value" in self.dataset_row:
                object.__setattr__(
                    self,
                    "input",
                    _make_read_only(self.dataset_row["attributes.input.value"]),
                )
            if "attributes.output.value" in self.dataset_row:
                object.__setattr__(
                    self,
                    "output",
                    _make_read_only(
                        self.dataset_row["attributes.output.value"]
                    ),
                )
            if "attributes.metadata" in self.dataset_row:
                object.__setattr__(
                    self,
                    "metadata",
                    _make_read_only(self.dataset_row["attributes.metadata"]),
                )
            if "id" in self.dataset_row:
                object.__setattr__(self, "id", self.dataset_row["id"])
            if "updated_at" in self.dataset_row:
                object.__setattr__(
                    self, "updated_at", self.dataset_row["updated_at"]
                )
        else:
            object.__setattr__(self, "input", self.input)
            object.__setattr__(self, "output", self.output)
            object.__setattr__(self, "metadata", self.metadata)

    @classmethod
    def from_dict(cls, obj: Mapping[str, object]) -> Example:
        """Create an Example instance from a dictionary."""
        return cls(
            id=cast("str", obj["id"]),
            input=cast("Mapping[str, JSONSerializable]", obj["input"]),
            output=cast("Mapping[str, JSONSerializable]", obj["output"]),
            metadata=cast(
                "Mapping[str, JSONSerializable]", obj.get("metadata") or {}
            ),
            updated_at=cast("datetime", obj["updated_at"]),
        )

    def __repr__(self) -> str:
        """Return a formatted string representation of the example."""
        spaces = " " * 4
        name = self.__class__.__name__
        identifiers = [f'{spaces}id="{self.id}",']
        contents = []
        for key in ("input", "output", "metadata", "dataset_row"):
            value = getattr(self, key, None)
            if value:
                contents.append(
                    spaces
                    + f"{_blue(key)}="
                    + json.dumps(
                        _shorten(value),
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=len(spaces),
                    )
                    .replace("\n", f"\n{spaces}")
                    .replace(' "..."\n', " ...\n")
                    + ","
                )
        return "\n".join([f"{name}(", *identifiers, *contents, ")"])


def _shorten(
    obj: dict[str, object] | list[object] | str | object, width: int = 50
) -> dict[str, object] | list[object] | str | object:
    if isinstance(obj, str):
        return textwrap.shorten(obj, width=width, placeholder="...")
    if isinstance(obj, dict):
        return {k: _shorten(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 2:
            return [_shorten(v) for v in obj[:2]] + ["..."]
        return [_shorten(v) for v in obj]
    return obj


def _make_read_only(
    obj: dict[str, object] | list[object] | str | object,
) -> dict[str, object] | list[object] | str | object:
    if isinstance(obj, dict):
        return _ReadOnly({k: _make_read_only(v) for k, v in obj.items()})
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return _ReadOnly(list(map(_make_read_only, obj)))
    return obj


class _ReadOnly(ObjectProxy):
    def __setitem__(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def __delitem__(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def __iadd__(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def pop(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def append(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def __copy__(self, *args: object, **kwargs: object) -> object:
        return copy(self.__wrapped__)

    def __deepcopy__(self, *args: object, **kwargs: object) -> object:
        return deepcopy(self.__wrapped__)

    def __repr__(self) -> str:
        return repr(self.__wrapped__)

    def __str__(self) -> str:
        return str(self.__wrapped__)


def _blue(text: str) -> str:
    return f"\033[1m\033[94m{text}\033[0m"


@dataclass(frozen=True)
class TestCase:
    """Container for an experiment test case with example data and repetition number."""

    example: Example
    repetition_number: RepetitionNumber


EXP_ID: ExperimentId = "EXP_ID"


def _exp_id() -> str:
    suffix = getrandbits(24).to_bytes(3, "big").hex()
    return f"{EXP_ID}_{suffix}"


@dataclass(frozen=True)
class ExperimentRun:
    """Represents a single run of an experiment.

    Args:
        start_time: The start time of the experiment run.
        end_time: The end time of the experiment run.
        experiment_id: The unique identifier for the experiment.
        dataset_example_id: The unique identifier for the dataset example.
        repetition_number: The repetition number of the experiment run.
        output: The output of the experiment run.
        error: The error message if the experiment run failed.
        id: The unique identifier for the experiment run.
        trace_id: The trace identifier for the experiment run.
    """

    start_time: datetime
    end_time: datetime
    experiment_id: ExperimentId
    dataset_example_id: ExampleId
    repetition_number: RepetitionNumber
    output: JSONSerializable
    error: str | None = None
    id: ExperimentRunId = field(default_factory=_exp_id)
    trace_id: TraceId | None = None

    @classmethod
    def from_dict(cls, obj: Mapping[str, object]) -> ExperimentRun:
        """Create an ExperimentRun instance from a dictionary."""
        return cls(
            start_time=cast("datetime", obj["start_time"]),
            end_time=cast("datetime", obj["end_time"]),
            experiment_id=cast("str", obj["experiment_id"]),
            dataset_example_id=cast("str", obj["dataset_example_id"]),
            repetition_number=cast("int", obj.get("repetition_number") or 1),
            output=cast("JSONSerializable", _make_read_only(obj.get("output"))),
            error=cast("str | None", obj.get("error")),
            id=cast("str", obj["id"]),
            trace_id=cast("str | None", obj.get("trace_id")),
        )

    def __post_init__(self) -> None:
        """Validate that exactly one of output or error is specified.

        Raises:
            ValueError: If both or neither output and error are specified.
        """
        if (self.output is None) == (self.error is None):
            raise ValueError(
                "Must specify exactly one of experiment_run_output or error"
            )


@dataclass(frozen=True)
class ExperimentEvaluationRun:
    """Represents a single evaluation run of an experiment.

    Args:
        experiment_run_id: The unique identifier for the experiment run.
        start_time: The start time of the evaluation run.
        end_time: The end time of the evaluation run.
        name: The name of the evaluation run.
        annotator_kind: The kind of annotator used in the evaluation run.
        error: The error message if the evaluation run failed.
        result (EvaluationResult | :obj:`None`): The result of the evaluation run.
        id (str): The unique identifier for the evaluation run.
        trace_id (TraceId | :obj:`None`): The trace identifier for the evaluation run.
    """

    experiment_run_id: ExperimentRunId
    start_time: datetime
    end_time: datetime
    name: str
    annotator_kind: str
    error: str | None = None
    result: EvaluationResult | None = None
    id: str = field(default_factory=_exp_id)
    trace_id: TraceId | None = None

    @classmethod
    def from_dict(cls, obj: Mapping[str, object]) -> ExperimentEvaluationRun:
        """Create an ExperimentEvaluationRun instance from a dictionary."""
        return cls(
            experiment_run_id=cast("str", obj["experiment_run_id"]),
            start_time=cast("datetime", obj["start_time"]),
            end_time=cast("datetime", obj["end_time"]),
            name=cast("str", obj["name"]),
            annotator_kind=cast("str", obj["annotator_kind"]),
            error=cast("str | None", obj.get("error")),
            result=EvaluationResult.from_dict(
                cast("Mapping[str, object] | None", obj.get("result"))
            ),
            id=cast("str", obj["id"]),
            trace_id=cast("str | None", obj.get("trace_id")),
        )

    def __post_init__(self) -> None:
        """Validate that exactly one of result or error is specified.

        Raises:
            ValueError: If both or neither result and error are specified.
        """
        if bool(self.result) == bool(self.error):
            raise ValueError("Must specify either result or error")


_LOCAL_TIMEZONE = datetime.now(timezone.utc).astimezone().tzinfo


def local_now() -> datetime:
    """Get the current datetime in the local timezone.

    Returns:
        A datetime object representing the current time in local timezone.
    """
    return datetime.now(timezone.utc).astimezone(tz=_LOCAL_TIMEZONE)


@dataclass(frozen=True)
class _HasStats:
    _title: str = field(repr=False, default="")
    _timestamp: datetime = field(repr=False, default_factory=local_now)
    stats: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def title(self) -> str:
        return f"{self._title} ({self._timestamp:%x %I:%M %p %z})"

    def __str__(self) -> str:
        try:
            pandas_major = int(version("pandas").split(".")[0])
            if pandas_major < 1:
                raise ImportError("Pandas version < 1.0")  # noqa: TRY301
            # `tabulate` is used by pandas >= 1.0 in DataFrame.to_markdown()
            import tabulate  # noqa: F401
        except ImportError:
            text = self.stats.__str__()
        else:
            text = self.stats.to_markdown(index=False)
        return f"{self.title}\n{'-' * len(self.title)}\n" + text


@dataclass(frozen=True)
class _TaskSummary(_HasStats):
    """Summary statistics of experiment task executions.

    **Users should not instantiate this object directly.**
    """

    _title: str = "Tasks Summary"

    @classmethod
    def from_task_runs(
        cls, n_examples: int, task_runs: Iterable[ExperimentRun | None]
    ) -> _TaskSummary:
        df = pd.DataFrame.from_records(
            [
                {
                    "example_id": run.dataset_example_id,
                    "error": run.error,
                }
                for run in task_runs
                if run is not None
            ]
        )
        n_runs = len(df)
        n_errors = 0 if df.empty else df.loc[:, "error"].astype(bool).sum()
        record = {
            "n_examples": n_examples,
            "n_runs": n_runs,
            "n_errors": n_errors,
            **(
                {"top_error": _top_string(df.loc[:, "error"])}
                if n_errors
                else {}
            ),
        }
        stats = pd.DataFrame.from_records([record])
        summary: _TaskSummary = object.__new__(cls)
        summary.__init__(stats=stats)  # type: ignore[misc]
        return summary

    @classmethod
    def __new__(cls, *args: object, **kwargs: object) -> NoReturn:
        # Direct instantiation by users is discouraged.
        raise NotImplementedError

    @classmethod
    def __init_subclass__(cls, **kwargs: object) -> None:
        # Direct sub-classing by users is discouraged.
        raise NotImplementedError


def _top_string(s: pd.Series, length: int = 100) -> str | None:
    if (cnt := s.dropna().str.slice(0, length).value_counts()).empty:
        return None
    return cast("str", cnt.sort_values(ascending=False).index[0])


@dataclass
class ExperimentTaskFieldNames:
    """Column names for mapping experiment task results in a :class:`pandas.DataFrame`.

    Args:
        example_id: Name of column containing example IDs.
            The ID values must match the id of the dataset rows.
        output: Name of column containing task results
    """

    example_id: str
    output: str


TaskOutput = JSONSerializable
ExampleOutput = Mapping[str, JSONSerializable]
ExampleMetadata = Mapping[str, JSONSerializable]
ExampleInput = Mapping[str, JSONSerializable]
ExperimentTask = (
    Callable[[Example], TaskOutput] | Callable[[Example], Awaitable[TaskOutput]]
)
