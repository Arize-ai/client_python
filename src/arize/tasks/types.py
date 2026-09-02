"""Public type re-exports for the tasks subdomain."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

from arize._generated.api_client.models.agent_call_run_config import (
    AgentCallRunConfig,
)
from arize._generated.api_client.models.agent_call_run_config_request import (
    AgentCallRunConfigRequest,
)
from arize._generated.api_client.models.list_task_runs_response import (
    ListTaskRunsResponse,
)
from arize._generated.api_client.models.llm_generation_run_config import (
    LlmGenerationRunConfig,
)
from arize._generated.api_client.models.llm_generation_run_config_request import (
    LlmGenerationRunConfigRequest,
)
from arize._generated.api_client.models.pagination_metadata import (
    PaginationMetadata,
)
from arize._generated.api_client.models.run_configuration import (
    RunConfiguration,
)
from arize._generated.api_client.models.run_configuration_request import (
    RunConfigurationRequest,
)
from arize._generated.api_client.models.task_evaluator import TaskEvaluator
from arize._generated.api_client.models.task_evaluator_input import (
    TaskEvaluatorInput,
)
from arize._generated.api_client.models.task_query_filter import (
    TaskQueryFilter,
)
from arize._generated.api_client.models.task_query_filters import (
    TaskQueryFilters,
)
from arize._generated.api_client.models.task_query_mapping import (
    TaskQueryMapping,
)
from arize._generated.api_client.models.task_run import TaskRun
from arize._generated.api_client.models.task_run_status import (
    TaskRunStatus as RunStatus,
)
from arize._generated.api_client.models.task_type import TaskType
from arize._generated.api_client.models.template_evaluation_run_config import (
    TemplateEvaluationRunConfig,
)
from arize._generated.api_client.models.template_evaluation_run_config_request import (
    TemplateEvaluationRunConfigRequest,
)


class Task(BaseModel):
    """SDK view of the generated ``Task`` with ``run_configuration`` unwrapped.

    The ``run_configuration`` field holds the concrete inner type
    (:class:`AgentCallRunConfig`, :class:`LlmGenerationRunConfig`, or
    :class:`TemplateEvaluationRunConfig`) instead of the oneOf wrapper
    :class:`RunConfiguration`.
    """

    id: str
    name: str
    type: str
    project_id: str | None = None
    dataset_id: str | None = None
    sampling_rate: float | None = None
    is_continuous: bool
    query_filter: str | None = None
    query_filters: TaskQueryFilters | None = None
    evaluators: list[TaskEvaluator]
    experiment_ids: list[str]
    run_configuration: (
        AgentCallRunConfig
        | LlmGenerationRunConfig
        | TemplateEvaluationRunConfig
        | None
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
    ) -> (
        AgentCallRunConfig
        | LlmGenerationRunConfig
        | TemplateEvaluationRunConfig
        | None
    ):
        if isinstance(v, RunConfiguration):
            if v.actual_instance is None:
                raise ValueError(
                    "RunConfiguration wrapper has actual_instance=None"
                )
            return v.actual_instance
        return v  # type: ignore[return-value]


class ListTasksResponse(BaseModel):
    """SDK view of the generated list response with each ``Task``'s ``run_configuration`` unwrapped."""

    tasks: list[Task]
    pagination: PaginationMetadata

    model_config = ConfigDict(from_attributes=True)


__all__ = [
    "AgentCallRunConfig",
    "AgentCallRunConfigRequest",
    "ListTaskRunsResponse",
    "ListTasksResponse",
    "LlmGenerationRunConfig",
    "LlmGenerationRunConfigRequest",
    "RunConfiguration",
    "RunConfigurationRequest",
    "RunStatus",
    "Task",
    "TaskEvaluator",
    "TaskEvaluatorInput",
    "TaskQueryFilter",
    "TaskQueryFilters",
    "TaskQueryMapping",
    "TaskRun",
    "TaskType",
    "TemplateEvaluationRunConfig",
    "TemplateEvaluationRunConfigRequest",
]
