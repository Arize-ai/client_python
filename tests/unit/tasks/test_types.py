"""Tests for arize.tasks.types public re-exports."""

from __future__ import annotations

from datetime import datetime

import pytest

import arize.tasks.types as types_module
from arize._generated.api_client.models.llm_generation_run_config import (
    LlmGenerationRunConfig,
)
from arize._generated.api_client.models.pagination_metadata import (
    PaginationMetadata,
)
from arize._generated.api_client.models.run_configuration import (
    RunConfiguration as _GenRunConfiguration,
)
from arize.tasks.types import (
    BaseEvaluationTaskRequestEvaluatorsInner,
    Task,
    TaskRun,
    TasksList200Response,
    TasksListRuns200Response,
    TemplateEvaluationRunConfig,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_llm_run_config() -> LlmGenerationRunConfig:
    return LlmGenerationRunConfig.model_construct(
        experiment_type="llm_generation",
        ai_integration_id="ai_integration_1",
        messages=[],
        input_variable_format=None,
    )


def _make_template_run_config() -> TemplateEvaluationRunConfig:
    return TemplateEvaluationRunConfig.model_construct(
        experiment_type="template_evaluation",
        ai_integration_id="ai_integration_1",
        template="evaluate {{input}}",
        provide_explanation=False,
    )


def _make_task(run_configuration: object = None) -> Task:
    return Task(
        id="t1",
        name="my-task",
        type="code_evaluation",
        project_id=None,
        dataset_id=None,
        sampling_rate=None,
        is_continuous=False,
        query_filter=None,
        evaluators=[],
        experiment_ids=[],
        run_configuration=run_configuration,
        last_run_at=None,
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
        created_by_user_id=None,
    )


@pytest.mark.unit
class TestTasksTypes:
    """Tests for the tasks types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        expected = {
            "BaseEvaluationTaskRequestEvaluatorsInner",
            "LlmGenerationRunConfig",
            "Task",
            "TaskRun",
            "TasksList200Response",
            "TasksListRuns200Response",
            "TemplateEvaluationRunConfig",
        }
        assert expected.issubset(set(types_module.__all__))

    @pytest.mark.parametrize(
        "cls",
        [
            BaseEvaluationTaskRequestEvaluatorsInner,
            LlmGenerationRunConfig,
            Task,
            TaskRun,
            TasksList200Response,
            TasksListRuns200Response,
            TemplateEvaluationRunConfig,
        ],
    )
    def test_type_is_class(self, cls: type) -> None:
        assert isinstance(cls, type)


@pytest.mark.unit
class TestTaskRunConfigurationCoercion:
    """Tests for Task._coerce_run_configuration validator."""

    def test_unwraps_llm_generation_run_config_from_wrapper(self) -> None:
        """RunConfiguration wrapping LlmGenerationRunConfig should be unwrapped."""
        llm_config = _make_llm_run_config()
        wrapper = _GenRunConfiguration.model_construct(
            actual_instance=llm_config
        )

        task = _make_task(run_configuration=wrapper)

        assert task.run_configuration is llm_config

    def test_unwraps_template_evaluation_run_config_from_wrapper(self) -> None:
        """RunConfiguration wrapping TemplateEvaluationRunConfig should be unwrapped."""
        tmpl_config = _make_template_run_config()
        wrapper = _GenRunConfiguration.model_construct(
            actual_instance=tmpl_config
        )

        task = _make_task(run_configuration=wrapper)

        assert task.run_configuration is tmpl_config

    def test_raises_when_wrapper_has_none_actual_instance(self) -> None:
        """RunConfiguration with actual_instance=None must raise ValueError."""
        null_wrapper = _GenRunConfiguration.model_construct(
            actual_instance=None
        )

        with pytest.raises(Exception, match="actual_instance=None"):
            _make_task(run_configuration=null_wrapper)

    def test_passes_through_llm_run_config_directly(self) -> None:
        """LlmGenerationRunConfig passed directly should not be transformed."""
        llm_config = _make_llm_run_config()

        task = _make_task(run_configuration=llm_config)

        assert task.run_configuration is llm_config

    def test_passes_through_template_run_config_directly(self) -> None:
        """TemplateEvaluationRunConfig passed directly should not be transformed."""
        tmpl_config = _make_template_run_config()

        task = _make_task(run_configuration=tmpl_config)

        assert task.run_configuration is tmpl_config

    def test_passes_through_none(self) -> None:
        """None run_configuration should remain None."""
        task = _make_task(run_configuration=None)

        assert task.run_configuration is None


@pytest.mark.unit
class TestTasksListCoercion:
    """Tests for TasksList200Response wrapping Task objects."""

    def _make_pagination(self) -> PaginationMetadata:
        return PaginationMetadata.model_construct(
            total_count=0, has_next_page=False, cursor=None
        )

    def test_tasks_list_preserves_run_configuration_unwrapping(self) -> None:
        """Tasks inside TasksList200Response should have their run_configuration unwrapped."""
        llm_config = _make_llm_run_config()
        wrapper = _GenRunConfiguration.model_construct(
            actual_instance=llm_config
        )
        task = _make_task(run_configuration=wrapper)

        response = TasksList200Response(
            tasks=[task],
            pagination=self._make_pagination(),
        )

        assert len(response.tasks) == 1
        assert response.tasks[0].run_configuration is llm_config

    def test_tasks_list_handles_none_run_configuration(self) -> None:
        """Tasks with no run_configuration should be stored as-is."""
        task = _make_task(run_configuration=None)

        response = TasksList200Response(
            tasks=[task],
            pagination=self._make_pagination(),
        )

        assert response.tasks[0].run_configuration is None

    def test_tasks_list_empty(self) -> None:
        """An empty task list should be preserved."""
        response = TasksList200Response(
            tasks=[],
            pagination=self._make_pagination(),
        )

        assert response.tasks == []
