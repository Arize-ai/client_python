"""Public type re-exports for the tasks subdomain."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator

from arize._generated.api_client.models.base_evaluation_task_request_evaluators_inner import (
    BaseEvaluationTaskRequestEvaluatorsInner,
)
from arize._generated.api_client.models.llm_generation_run_config import (
    LlmGenerationRunConfig,
)
from arize._generated.api_client.models.pagination_metadata import (
    PaginationMetadata,
)
from arize._generated.api_client.models.run_configuration import (
    RunConfiguration as _GenRunConfiguration,
)
from arize._generated.api_client.models.task_evaluator import TaskEvaluator
from arize._generated.api_client.models.task_run import TaskRun
from arize._generated.api_client.models.tasks_list_runs200_response import (
    TasksListRuns200Response,
)
from arize._generated.api_client.models.template_evaluation_run_config import (
    TemplateEvaluationRunConfig,
)


class TaskType(str, Enum):
    """Task type discriminator values, mirroring the generated OpenAPI schema."""

    TEMPLATE_EVALUATION = "template_evaluation"
    CODE_EVALUATION = "code_evaluation"
    RUN_EXPERIMENT = "run_experiment"


class RunStatus(str, Enum):
    """Task run status values, mirroring the generated OpenAPI schema."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """SDK view of the generated ``Task`` with ``run_configuration`` unwrapped.

    The ``run_configuration`` field holds the concrete inner type
    (:class:`LlmGenerationRunConfig` or :class:`TemplateEvaluationRunConfig`)
    instead of the oneOf wrapper :class:`RunConfiguration`.
    """

    id: str
    name: str
    type: str
    project_id: str | None = None
    dataset_id: str | None = None
    sampling_rate: float | None = None
    is_continuous: bool
    query_filter: str | None = None
    evaluators: list[TaskEvaluator]
    experiment_ids: list[str]
    run_configuration: (
        LlmGenerationRunConfig | TemplateEvaluationRunConfig | None
    ) = None
    last_run_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    created_by_user_id: str | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("run_configuration", mode="before")
    @classmethod
    def _coerce_run_configuration(
        cls, v: object
    ) -> LlmGenerationRunConfig | TemplateEvaluationRunConfig | None:
        if isinstance(v, _GenRunConfiguration):
            if v.actual_instance is None:
                raise ValueError(
                    "RunConfiguration wrapper has actual_instance=None"
                )
            return v.actual_instance
        return v  # type: ignore[return-value]


class TasksList200Response(BaseModel):
    """SDK view of the generated list response with each ``Task``'s ``run_configuration`` unwrapped."""

    tasks: list[Task]
    pagination: PaginationMetadata

    model_config = ConfigDict(from_attributes=True)


__all__ = [
    "BaseEvaluationTaskRequestEvaluatorsInner",
    "LlmGenerationRunConfig",
    "RunStatus",
    "Task",
    "TaskEvaluator",
    "TaskRun",
    "TaskType",
    "TasksList200Response",
    "TasksListRuns200Response",
    "TemplateEvaluationRunConfig",
]
