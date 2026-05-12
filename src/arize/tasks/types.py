"""Public type re-exports for the tasks subdomain."""

from enum import Enum

from arize._generated.api_client.models.base_evaluation_task_request_evaluators_inner import (
    BaseEvaluationTaskRequestEvaluatorsInner,
)
from arize._generated.api_client.models.llm_generation_run_config import (
    LlmGenerationRunConfig,
)
from arize._generated.api_client.models.run_configuration import (
    RunConfiguration,
)
from arize._generated.api_client.models.task import Task
from arize._generated.api_client.models.task_run import TaskRun
from arize._generated.api_client.models.tasks_list200_response import (
    TasksList200Response,
)
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


__all__ = [
    "BaseEvaluationTaskRequestEvaluatorsInner",
    "LlmGenerationRunConfig",
    "RunConfiguration",
    "RunStatus",
    "Task",
    "TaskRun",
    "TaskType",
    "TasksList200Response",
    "TasksListRuns200Response",
    "TemplateEvaluationRunConfig",
]
