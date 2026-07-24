"""Integration coverage for TasksClient validation against the real Arize API.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_tasks_flows.py -m integration -v
"""

from __future__ import annotations

import contextlib
import os
import uuid
from typing import Any

import pytest

from arize._generated.api_client.exceptions import ApiException
from arize.tasks.types import TaskEvaluatorInput, TaskType

API_KEY = os.environ.get("ARIZE_API_KEY", "")
SPACE_NAME = os.environ.get("ARIZE_TEST_SPACE_NAME", "")

CUSTOM_CODE_EVALUATOR_LIMIT_MESSAGE = (
    "Only one custom code evaluator runs per task. "
    "Create one task per custom evaluator."
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY or not SPACE_NAME,
        reason="ARIZE_API_KEY and ARIZE_TEST_SPACE_NAME must be set",
    ),
]


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _custom_code_config(name: str) -> Any:
    from arize._generated import api_client as gen

    return gen.CustomCodeConfig(
        type="CUSTOM",
        name=name,
        code=(
            "class IntegrationEvaluator(CodeEvaluator):\n"
            "    def evaluate(self, output: str, **kwargs) -> EvaluationResult:\n"
            "        return EvaluationResult(score=1.0)\n"
        ),
        variables=["output"],
    )


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


class TestTasksCustomCodeEvaluatorLimit:
    """The REST API rejects multiple custom code evaluators for one task."""

    def test_create_and_update_reject_two_custom_code_evaluators(
        self, arize_client: Any
    ) -> None:
        project = arize_client.projects.create(
            name=_unique("sdk-test-task-project"),
            space=SPACE_NAME,
        )
        evaluator_one = None
        evaluator_two = None
        task = None
        try:
            evaluator_one = arize_client.evaluators.create_code_evaluator(
                name=_unique("sdk-test-custom-evaluator"),
                space=SPACE_NAME,
                commit_message="initial version",
                code_config=_custom_code_config("custom_score_one"),
            )
            evaluator_two = arize_client.evaluators.create_code_evaluator(
                name=_unique("sdk-test-custom-evaluator"),
                space=SPACE_NAME,
                commit_message="initial version",
                code_config=_custom_code_config("custom_score_two"),
            )
            evaluator_inputs = [
                TaskEvaluatorInput(evaluator_id=evaluator_one.id),
                TaskEvaluatorInput(evaluator_id=evaluator_two.id),
            ]

            with pytest.raises(ApiException) as create_error:
                arize_client.tasks.create_evaluation_task(
                    name=_unique("sdk-test-two-custom-evaluators"),
                    task_type=TaskType.CODE_EVALUATION,
                    evaluators=evaluator_inputs,
                    project=project.id,
                )
            assert create_error.value.status == 422
            assert CUSTOM_CODE_EVALUATOR_LIMIT_MESSAGE in str(
                create_error.value
            )

            task = arize_client.tasks.create_evaluation_task(
                name=_unique("sdk-test-one-custom-evaluator"),
                task_type=TaskType.CODE_EVALUATION,
                evaluators=[TaskEvaluatorInput(evaluator_id=evaluator_one.id)],
                project=project.id,
            )
            with pytest.raises(ApiException) as update_error:
                arize_client.tasks.update(
                    task=task.id,
                    evaluators=evaluator_inputs,
                )
            assert update_error.value.status == 422
            assert CUSTOM_CODE_EVALUATOR_LIMIT_MESSAGE in str(
                update_error.value
            )
        finally:
            if task is not None:
                with contextlib.suppress(ApiException):
                    arize_client.tasks.delete(task=task.id)
            if evaluator_two is not None:
                with contextlib.suppress(ApiException):
                    arize_client.evaluators.delete(evaluator=evaluator_two.id)
            if evaluator_one is not None:
                with contextlib.suppress(ApiException):
                    arize_client.evaluators.delete(evaluator=evaluator_one.id)
            with contextlib.suppress(ApiException):
                arize_client.projects.delete(project=project.id)
