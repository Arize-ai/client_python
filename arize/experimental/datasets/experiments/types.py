import copy
import json
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import version
from random import getrandbits
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import pandas as pd
from typing_extensions import TypeAlias
from wrapt import ObjectProxy


class AnnotatorKind(Enum):
    CODE = "CODE"
    LLM = "LLM"


JSONSerializable: TypeAlias = Optional[Union[Dict[str, Any], List[Any], str, int, float, bool]]
Score: TypeAlias = Optional[Union[bool, int, float]]
Label: TypeAlias = Optional[str]
Explanation: TypeAlias = Optional[str]

EvaluatorName: TypeAlias = str
EvaluatorKind: TypeAlias = str
EvaluatorOutput: TypeAlias = Union[
    "EvaluationResult", bool, int, float, str, Tuple[Score, Label, Explanation]
]

ExperimentId: TypeAlias = str
DatasetId: TypeAlias = str
DatasetVersionId: TypeAlias = str
ExampleId: TypeAlias = str
RepetitionNumber: TypeAlias = int
ExperimentRunId: TypeAlias = str
TraceId: TypeAlias = str


@dataclass(frozen=True)
class Example:
    id: ExampleId = field(default_factory=str)
    updated_at: datetime = field(default_factory=datetime.now)
    input: Mapping[str, JSONSerializable] = field(default_factory=dict)
    output: Mapping[str, JSONSerializable] = field(default_factory=dict)
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)
    dataset_row: Mapping[str, JSONSerializable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dataset_row is not None:
            object.__setattr__(self, "dataset_row", _make_read_only(self.dataset_row))
            if "attributes.input.value" in self.dataset_row.keys():
                object.__setattr__(
                    self,
                    "input",
                    _make_read_only(self.dataset_row["attributes.input.value"]),
                )
            if "attributes.output.value" in self.dataset_row.keys():
                object.__setattr__(
                    self,
                    "output",
                    _make_read_only(self.dataset_row["attributes.output.value"]),
                )
            if "metadata" in self.dataset_row.keys():
                object.__setattr__(self, "metadata", _make_read_only(self.dataset_row["metadata"]))
            if "id" in self.dataset_row.keys():
                object.__setattr__(self, "id", self.dataset_row["id"])
            if "updated_at" in self.dataset_row.keys():
                object.__setattr__(self, "updated_at", self.dataset_row["updated_at"])
        else:
            object.__setattr__(self, "input", self.input)
            object.__setattr__(self, "output", self.output)
            object.__setattr__(self, "metadata", self.metadata)

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "Example":
        return cls(
            input=obj["input"],
            output=obj["output"],
            metadata=obj.get("metadata") or {},
            id=obj["id"],
            updated_at=obj["updated_at"],
        )

    def __repr__(self) -> str:
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


@dataclass(frozen=True)
class EvaluationResult:
    score: Optional[float] = None
    label: Optional[str] = None
    explanation: Optional[str] = None
    metadata: Mapping[str, JSONSerializable] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, obj: Optional[Mapping[str, Any]]) -> Optional["EvaluationResult"]:
        if not obj:
            return None
        return cls(
            score=obj.get("score"),
            label=obj.get("label"),
            explanation=obj.get("explanation"),
            metadata=obj.get("metadata") or {},
        )

    def __post_init__(self) -> None:
        if self.score is None and not self.label:
            raise ValueError("Must specify score or label, or both")
        if self.score is None and not self.label:
            object.__setattr__(self, "score", 0)
        for k in ("label", "explanation"):
            v = getattr(self, k, None)
            if v is not None:
                object.__setattr__(self, k, str(v) or None)


TaskOutput: TypeAlias = JSONSerializable
ExampleOutput: TypeAlias = Mapping[str, JSONSerializable]
ExampleMetadata: TypeAlias = Mapping[str, JSONSerializable]
ExampleInput: TypeAlias = Mapping[str, JSONSerializable]
ExperimentTask: TypeAlias = Union[
    Callable[[Example], TaskOutput],
    Callable[[Example], Awaitable[TaskOutput]],
]

T = TypeVar("T")


def _shorten(obj: Any, width: int = 50) -> Any:
    if isinstance(obj, str):
        return textwrap.shorten(obj, width=width, placeholder="...")
    if isinstance(obj, dict):
        return {k: _shorten(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 2:
            return [_shorten(v) for v in obj[:2]] + ["..."]
        return [_shorten(v) for v in obj]
    return obj


def _make_read_only(obj: Any) -> Any:
    if isinstance(obj, dict):
        return _ReadOnly({k: _make_read_only(v) for k, v in obj.items()})
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return _ReadOnly(list(map(_make_read_only, obj)))
    return obj


class _ReadOnly(ObjectProxy):  # type: ignore[misc]
    def __setitem__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __delitem__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __iadd__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def append(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __copy__(self, *args: Any, **kwargs: Any) -> Any:
        return copy(self.__wrapped__)

    def __deepcopy__(self, *args: Any, **kwargs: Any) -> Any:
        return deepcopy(self.__wrapped__)

    def __repr__(self) -> str:
        return repr(self.__wrapped__)

    def __str__(self) -> str:
        return str(self.__wrapped__)


def _blue(text: str) -> str:
    return f"\033[1m\033[94m{text}\033[0m"


@dataclass(frozen=True)
class TestCase:
    example: Example
    repetition_number: "RepetitionNumber"


EXP_ID: ExperimentId = "EXP_ID"


def _exp_id() -> str:
    suffix = getrandbits(24).to_bytes(3, "big").hex()
    return f"{EXP_ID}_{suffix}"


@dataclass(frozen=True)
class ExperimentRun:
    start_time: datetime
    end_time: datetime
    experiment_id: ExperimentId
    dataset_example_id: ExampleId
    repetition_number: RepetitionNumber
    output: JSONSerializable
    error: Optional[str] = None
    id: ExperimentRunId = field(default_factory=_exp_id)
    trace_id: Optional[TraceId] = None

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "ExperimentRun":
        return cls(
            start_time=obj["start_time"],
            end_time=obj["end_time"],
            experiment_id=obj["experiment_id"],
            dataset_example_id=obj["dataset_example_id"],
            repetition_number=obj.get("repetition_number") or 1,
            output=_make_read_only(obj.get("output")),
            error=obj.get("error"),
            id=obj["id"],
            trace_id=obj.get("trace_id"),
        )

    def __post_init__(self) -> None:
        if bool(self.output) == bool(self.error):
            raise ValueError("Must specify exactly one of experiment_run_output or error")


@dataclass(frozen=True)
class ExperimentEvaluationRun:
    experiment_run_id: ExperimentRunId
    start_time: datetime
    end_time: datetime
    name: str
    annotator_kind: str
    error: Optional[str] = None
    result: Optional[EvaluationResult] = None
    id: str = field(default_factory=_exp_id)
    trace_id: Optional[TraceId] = None

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "ExperimentEvaluationRun":
        return cls(
            experiment_run_id=obj["experiment_run_id"],
            start_time=obj["start_time"],
            end_time=obj["end_time"],
            name=obj["name"],
            annotator_kind=obj["annotator_kind"],
            error=obj.get("error"),
            result=EvaluationResult.from_dict(obj.get("result")),
            id=obj["id"],
            trace_id=obj.get("trace_id"),
        )

    def __post_init__(self) -> None:
        if bool(self.result) == bool(self.error):
            raise ValueError("Must specify either result or error")


_LOCAL_TIMEZONE = datetime.now(timezone.utc).astimezone().tzinfo


def local_now() -> datetime:
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
            assert int(version("pandas").split(".")[0]) >= 1
            # `tabulate` is used by pandas >= 1.0 in DataFrame.to_markdown()
            import tabulate  # noqa: F401
        except (AssertionError, ImportError):
            text = self.stats.__str__()
        else:
            text = self.stats.to_markdown(index=False)
        return f"{self.title}\n{'-' * len(self.title)}\n" + text


@dataclass(frozen=True)
class _TaskSummary(_HasStats):
    """
    Summary statistics of experiment task executions.

    **Users should not instantiate this object directly.**
    """

    _title: str = "Tasks Summary"

    @classmethod
    def from_task_runs(
        cls,
        n_examples: int,
        task_runs: Iterable[Optional[ExperimentRun]],
    ) -> "_TaskSummary":
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
            **(dict(top_error=_top_string(df.loc[:, "error"])) if n_errors else {}),
        }
        stats = pd.DataFrame.from_records([record])
        summary: _TaskSummary = object.__new__(cls)
        summary.__init__(stats=stats)  # type: ignore[misc]
        return summary

    @classmethod
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        # Direct instantiation by users is discouraged.
        raise NotImplementedError

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Direct sub-classing by users is discouraged.
        raise NotImplementedError


def _top_string(s: "pd.Series[Any]", length: int = 100) -> Optional[str]:
    if (cnt := s.dropna().str.slice(0, length).value_counts()).empty:
        return None
    return cast(str, cnt.sort_values(ascending=False).index[0])
