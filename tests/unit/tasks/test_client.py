"""Unit tests for src/arize/tasks/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize._generated.api_client.models.run_configuration import (
    RunConfiguration,
)
from arize.tasks.client import (
    _DEFAULT_POLL_INTERVAL,
    _DEFAULT_TIMEOUT,
    TasksClient,
)
from arize.tasks.types import Task, TasksList200Response

# Base64 IDs that pass _is_resource_id() — decode to "Type:123"
_TASK_ID = "VGFzazoxMjM="  # Task:123
_PROJECT_ID = "UHJvamVjdDoxMjM="  # Project:123
_DATASET_ID = "RGF0YXNldDoxMjM="  # Dataset:123


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock TasksApi instance."""
    return Mock()


@pytest.fixture
def tasks_client(mock_sdk_config: Mock, mock_api: Mock) -> TasksClient:
    """Provide a TasksClient with mocked internals."""
    with patch("arize._generated.api_client.TasksApi", return_value=mock_api):
        return TasksClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestTasksClientInit:
    """Tests for TasksClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.TasksApi", return_value=mock_api
        ):
            client = TasksClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_tasks_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to TasksApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.TasksApi"
        ) as mock_tasks_api_cls:
            TasksClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_tasks_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestTasksClientList:
    """Tests for TasksClient.list()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            TasksList200Response,
            "model_validate",
            side_effect=lambda v, **kw: v,
        ):
            yield

    def test_list_with_space_id(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """list() should resolve project/dataset IDs and space ID correctly."""
        tasks_client.list(
            name="my-task",
            space="U3BhY2U6OTA1MDoxSmtS",
            project=_PROJECT_ID,
            dataset=_DATASET_ID,
            task_type="template_evaluation",
            limit=50,
            cursor="cursor-xyz",
        )

        mock_api.tasks_list.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS",
            space_name=None,
            name="my-task",
            project_id=_PROJECT_ID,
            dataset_id=_DATASET_ID,
            type="template_evaluation",
            limit=50,
            cursor="cursor-xyz",
        )

    def test_list_with_space_name(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """list() should resolve a non-prefixed space value to space_name."""
        tasks_client.list(
            name="my-task",
            space="my-space",
        )

        mock_api.tasks_list.assert_called_once_with(
            space_id=None,
            space_name="my-space",
            name="my-task",
            project_id=None,
            dataset_id=None,
            type=None,
            limit=100,
            cursor=None,
        )

    def test_list_with_project_name_resolves_id(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """list() should resolve a project name to an ID via ProjectsApi."""
        mock_project = Mock()
        mock_project.id = _PROJECT_ID
        mock_project.name = "my-project"
        mock_projects_api = Mock()
        mock_projects_api.projects_list.return_value = Mock(
            projects=[mock_project],
            pagination=Mock(next_cursor=None),
        )
        tasks_client._projects_api = mock_projects_api

        tasks_client.list(project="my-project", space="U3BhY2U6OTA1MDoxSmtS")

        mock_api.tasks_list.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS",
            space_name=None,
            name=None,
            project_id=_PROJECT_ID,
            dataset_id=None,
            type=None,
            limit=100,
            cursor=None,
        )

    def test_list_defaults(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """list() should default all filters to None and limit to 100."""
        tasks_client.list()

        mock_api.tasks_list.assert_called_once_with(
            space_id=None,
            space_name=None,
            name=None,
            project_id=None,
            dataset_id=None,
            type=None,
            limit=100,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from tasks_list."""
        expected = Mock()
        mock_api.tasks_list.return_value = expected

        result = tasks_client.list()

        assert result is expected

    def test_list_emits_alpha_prerelease_warning(
        self,
        tasks_client: TasksClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        tasks_client.list()

        assert any(
            "ALPHA" in record.message and "tasks.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestTasksClientGet:
    """Tests for TasksClient.get()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            Task, "model_validate", side_effect=lambda v, **kw: v
        ):
            yield

    def test_get_calls_api_with_task_id(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """get() should resolve task and forward task_id to tasks_get."""
        tasks_client.get(task=_TASK_ID)

        mock_api.tasks_get.assert_called_once_with(task_id=_TASK_ID)

    def test_get_returns_api_response(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """get() should propagate the return value from tasks_get."""
        expected = Mock()
        mock_api.tasks_get.return_value = expected

        result = tasks_client.get(task=_TASK_ID)

        assert result is expected

    def test_get_emits_alpha_prerelease_warning(
        self,
        tasks_client: TasksClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        tasks_client.get(task=_TASK_ID)

        assert any(
            "ALPHA" in record.message and "tasks.get" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestTasksClientCreate:
    """Tests for TasksClient.create()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            Task, "model_validate", side_effect=lambda v, **kw: v
        ):
            yield

    def test_create_builds_request_and_calls_api(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create() should build the inner request, wrap it, and call tasks_create."""
        mock_evaluator = Mock()

        with (
            patch(
                "arize._generated.api_client.CreateTemplateEvaluationTaskRequest"
            ) as mock_inner_cls,
            patch(
                "arize._generated.api_client.TasksCreateRequest"
            ) as mock_wrapper_cls,
        ):
            mock_inner = Mock()
            mock_inner_cls.return_value = mock_inner
            mock_wrapper = Mock()
            mock_wrapper_cls.return_value = mock_wrapper

            tasks_client._create(
                name="my-task",
                task_type="template_evaluation",
                evaluators=[mock_evaluator],
                project=_PROJECT_ID,
            )

        mock_inner_cls.assert_called_once_with(
            name="my-task",
            type="template_evaluation",
            evaluators=[mock_evaluator],
            project_id=_PROJECT_ID,
            dataset_id=None,
            experiment_ids=None,
            sampling_rate=None,
            is_continuous=None,
            query_filter=None,
        )
        mock_wrapper_cls.assert_called_once_with(actual_instance=mock_inner)
        mock_api.tasks_create.assert_called_once_with(
            tasks_create_request=mock_wrapper
        )

    def test_create_code_evaluation_uses_code_eval_inner(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """code_evaluation task_type should construct CreateCodeEvaluationTaskRequest."""
        with (
            patch(
                "arize._generated.api_client.CreateCodeEvaluationTaskRequest"
            ) as mock_inner_cls,
            patch("arize._generated.api_client.TasksCreateRequest"),
        ):
            mock_inner_cls.return_value = Mock()

            tasks_client._create(
                name="ds-task",
                task_type="code_evaluation",
                evaluators=[Mock()],
                dataset=_DATASET_ID,
                experiment_ids=["exp-1", "exp-2"],
            )

        _, kwargs = mock_inner_cls.call_args
        assert kwargs["type"] == "code_evaluation"
        assert kwargs["dataset_id"] == _DATASET_ID
        assert kwargs["experiment_ids"] == ["exp-1", "exp-2"]
        assert kwargs["project_id"] is None

    def test_create_with_optional_fields(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create() should forward optional sampling_rate, is_continuous, query_filter."""
        with (
            patch(
                "arize._generated.api_client.CreateTemplateEvaluationTaskRequest"
            ) as mock_inner_cls,
            patch("arize._generated.api_client.TasksCreateRequest"),
        ):
            mock_inner_cls.return_value = Mock()

            tasks_client._create(
                name="my-task",
                task_type="template_evaluation",
                evaluators=[Mock()],
                project=_PROJECT_ID,
                sampling_rate=0.5,
                is_continuous=True,
                query_filter="span_kind == 'LLM'",
            )

        _, kwargs = mock_inner_cls.call_args
        assert kwargs["sampling_rate"] == 0.5
        assert kwargs["is_continuous"] is True
        assert kwargs["query_filter"] == "span_kind == 'LLM'"

    def test_create_run_experiment_builds_correct_request(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create(task_type='run_experiment') should use CreateRunExperimentTaskRequest."""
        mock_run_config = Mock(spec=RunConfiguration)

        with (
            patch(
                "arize._generated.api_client.CreateRunExperimentTaskRequest"
            ) as mock_inner_cls,
            patch(
                "arize._generated.api_client.TasksCreateRequest"
            ) as mock_wrapper_cls,
        ):
            mock_inner = Mock()
            mock_inner_cls.return_value = mock_inner
            mock_wrapper = Mock()
            mock_wrapper_cls.return_value = mock_wrapper

            tasks_client._create(
                name="exp-task",
                task_type="run_experiment",
                run_configuration=mock_run_config,
                dataset=_DATASET_ID,
            )

        mock_inner_cls.assert_called_once_with(
            name="exp-task",
            type="run_experiment",
            dataset_id=_DATASET_ID,
            run_configuration=mock_run_config,
        )
        mock_wrapper_cls.assert_called_once_with(actual_instance=mock_inner)
        mock_api.tasks_create.assert_called_once_with(
            tasks_create_request=mock_wrapper
        )

    def test_create_run_experiment_rejects_eval_only_fields(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create(task_type='run_experiment') should raise if eval-only fields are set."""
        with pytest.raises(ValueError, match="eval-only fields"):
            tasks_client._create(
                name="bad",
                task_type="run_experiment",
                evaluators=[Mock()],
                run_configuration=Mock(spec=RunConfiguration),
                dataset=_DATASET_ID,
            )

        mock_api.tasks_create.assert_not_called()

    def test_create_run_experiment_requires_run_configuration(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create(task_type='run_experiment') should raise when run_configuration is absent."""
        with pytest.raises(ValueError, match="run_configuration"):
            tasks_client._create(
                name="bad",
                task_type="run_experiment",
                dataset=_DATASET_ID,
            )

        mock_api.tasks_create.assert_not_called()

    def test_create_eval_rejects_run_configuration(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create() for eval types should raise when run_configuration is provided."""
        with pytest.raises(ValueError, match="run_configuration"):
            tasks_client._create(
                name="bad",
                task_type="template_evaluation",
                evaluators=[Mock()],
                run_configuration=Mock(spec=RunConfiguration),
                project=_PROJECT_ID,
            )

        mock_api.tasks_create.assert_not_called()

    def test_create_returns_api_response(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from tasks_create."""
        expected = Mock()
        mock_api.tasks_create.return_value = expected

        with (
            patch(
                "arize._generated.api_client.CreateTemplateEvaluationTaskRequest"
            ),
            patch("arize._generated.api_client.TasksCreateRequest"),
        ):
            result = tasks_client._create(
                name="my-task",
                task_type="template_evaluation",
                evaluators=[Mock()],
                project=_PROJECT_ID,
            )

        assert result is expected


@pytest.mark.unit
class TestTasksClientCreateEvaluationTask:
    """Tests for TasksClient.create_evaluation_task()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            Task, "model_validate", side_effect=lambda v, **kw: v
        ):
            yield

    def test_delegates_template_evaluation_to_create(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create_evaluation_task() with template_evaluation should call create()."""
        mock_evaluator = Mock()

        with (
            patch(
                "arize._generated.api_client.CreateTemplateEvaluationTaskRequest"
            ) as mock_inner_cls,
            patch("arize._generated.api_client.TasksCreateRequest"),
        ):
            mock_inner_cls.return_value = Mock()
            tasks_client.create_evaluation_task(
                name="my-task",
                task_type="template_evaluation",
                evaluators=[mock_evaluator],
                project=_PROJECT_ID,
            )

        _, kwargs = mock_inner_cls.call_args
        assert kwargs["type"] == "template_evaluation"
        assert kwargs["project_id"] == _PROJECT_ID
        mock_api.tasks_create.assert_called_once()

    def test_delegates_code_evaluation_to_create(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create_evaluation_task() with code_evaluation should call create()."""
        with (
            patch(
                "arize._generated.api_client.CreateCodeEvaluationTaskRequest"
            ) as mock_inner_cls,
            patch("arize._generated.api_client.TasksCreateRequest"),
        ):
            mock_inner_cls.return_value = Mock()
            tasks_client.create_evaluation_task(
                name="code-task",
                task_type="code_evaluation",
                evaluators=[Mock()],
                dataset=_DATASET_ID,
                experiment_ids=["exp-1"],
            )

        _, kwargs = mock_inner_cls.call_args
        assert kwargs["type"] == "code_evaluation"
        assert kwargs["dataset_id"] == _DATASET_ID
        assert kwargs["experiment_ids"] == ["exp-1"]

    def test_returns_api_response(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create_evaluation_task() should return the task from the API."""
        expected = Mock()
        mock_api.tasks_create.return_value = expected

        with (
            patch(
                "arize._generated.api_client.CreateTemplateEvaluationTaskRequest"
            ),
            patch("arize._generated.api_client.TasksCreateRequest"),
        ):
            result = tasks_client.create_evaluation_task(
                name="my-task",
                task_type="template_evaluation",
                evaluators=[Mock()],
                project=_PROJECT_ID,
            )

        assert result is expected


@pytest.mark.unit
class TestTasksClientCreateRunExperimentTask:
    """Tests for TasksClient.create_run_experiment_task()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            Task, "model_validate", side_effect=lambda v, **kw: v
        ):
            yield

    def test_delegates_to_create(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create_run_experiment_task() should call create() with run_experiment type."""
        mock_run_config = Mock(spec=RunConfiguration)

        with (
            patch(
                "arize._generated.api_client.RunConfiguration"
            ) as mock_rc_cls,
            patch(
                "arize._generated.api_client.CreateRunExperimentTaskRequest"
            ) as mock_inner_cls,
            patch("arize._generated.api_client.TasksCreateRequest"),
        ):
            mock_rc_cls.return_value = Mock()
            mock_inner_cls.return_value = Mock()
            tasks_client.create_run_experiment_task(
                name="run-exp-task",
                dataset=_DATASET_ID,
                run_configuration=mock_run_config,
            )

        _, kwargs = mock_inner_cls.call_args
        assert kwargs["type"] == "run_experiment"
        assert kwargs["dataset_id"] == _DATASET_ID
        mock_api.tasks_create.assert_called_once()

    def test_forwards_space_for_dataset_resolution(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create_run_experiment_task() should pass space through to create()."""
        mock_dataset = Mock()
        mock_dataset.id = _DATASET_ID
        mock_dataset.name = "my-dataset"
        mock_datasets_api = Mock()
        mock_datasets_api.datasets_list.return_value = Mock(
            datasets=[mock_dataset],
            pagination=Mock(next_cursor=None),
        )
        tasks_client._datasets_api = mock_datasets_api

        with (
            patch("arize._generated.api_client.RunConfiguration"),
            patch(
                "arize._generated.api_client.CreateRunExperimentTaskRequest"
            ) as mock_inner_cls,
            patch("arize._generated.api_client.TasksCreateRequest"),
        ):
            mock_inner_cls.return_value = Mock()
            tasks_client.create_run_experiment_task(
                name="run-exp-task",
                dataset="my-dataset",
                run_configuration=Mock(spec=RunConfiguration),
                space="U3BhY2U6OTA1MDoxSmtS",
            )

        _, kwargs = mock_inner_cls.call_args
        assert kwargs["dataset_id"] == _DATASET_ID

    def test_returns_api_response(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create_run_experiment_task() should return the task from the API."""
        expected = Mock()
        mock_api.tasks_create.return_value = expected

        with (
            patch("arize._generated.api_client.RunConfiguration"),
            patch("arize._generated.api_client.CreateRunExperimentTaskRequest"),
            patch("arize._generated.api_client.TasksCreateRequest"),
        ):
            result = tasks_client.create_run_experiment_task(
                name="run-exp-task",
                dataset=_DATASET_ID,
                run_configuration=Mock(spec=RunConfiguration),
            )

        assert result is expected


@pytest.mark.unit
class TestTasksClientUpdate:
    """Tests for TasksClient.update()."""

    @pytest.fixture(autouse=True)
    def _bypass_model_validate(self) -> None:
        with patch.object(
            Task, "model_validate", side_effect=lambda v, **kw: v
        ):
            yield

    def test_update_builds_request_and_calls_api(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """update() should build the inner request, wrap it, and call tasks_update."""
        with (
            patch(
                "arize._generated.api_client.UpdateEvaluationTaskRequest"
            ) as mock_inner_cls,
            patch(
                "arize._generated.api_client.TasksUpdateRequest"
            ) as mock_wrapper_cls,
        ):
            mock_inner = Mock()
            mock_inner_cls.return_value = mock_inner
            mock_wrapper = Mock()
            mock_wrapper_cls.return_value = mock_wrapper

            tasks_client.update(task=_TASK_ID, name="new-name")

        mock_inner_cls.assert_called_once_with(name="new-name")
        mock_wrapper_cls.assert_called_once_with(actual_instance=mock_inner)
        mock_api.tasks_update.assert_called_once_with(
            task_id=_TASK_ID,
            tasks_update_request=mock_wrapper,
        )

    def test_update_with_all_optional_fields(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """update() should forward all mutable fields."""
        mock_ev = Mock()
        with (
            patch(
                "arize._generated.api_client.UpdateEvaluationTaskRequest"
            ) as mock_inner_cls,
            patch("arize._generated.api_client.TasksUpdateRequest"),
        ):
            mock_inner_cls.return_value = Mock()

            tasks_client.update(
                task=_TASK_ID,
                space="my-space",
                name="x",
                sampling_rate=0.25,
                is_continuous=True,
                query_filter="span_kind == 'LLM'",
                evaluators=[mock_ev],
            )

        _, kwargs = mock_inner_cls.call_args
        assert kwargs["name"] == "x"
        assert kwargs["sampling_rate"] == 0.25
        assert kwargs["is_continuous"] is True
        assert kwargs["query_filter"] == "span_kind == 'LLM'"
        assert kwargs["evaluators"] == [mock_ev]

    def test_update_with_query_filter_none(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """Passing query_filter=None should clear the filter on the API."""
        with (
            patch(
                "arize._generated.api_client.UpdateEvaluationTaskRequest"
            ) as mock_inner_cls,
            patch("arize._generated.api_client.TasksUpdateRequest"),
        ):
            mock_inner_cls.return_value = Mock()

            tasks_client.update(task=_TASK_ID, query_filter=None)

        mock_inner_cls.assert_called_once_with(query_filter=None)

    def test_update_raises_when_no_fields(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """update() should reject an empty patch."""
        with pytest.raises(ValueError, match="At least one update field"):
            tasks_client.update(task=_TASK_ID)

        mock_api.tasks_update.assert_not_called()

    def test_update_returns_api_response(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """update() should propagate the return value from tasks_update."""
        expected = Mock()
        mock_api.tasks_update.return_value = expected

        with (
            patch("arize._generated.api_client.UpdateEvaluationTaskRequest"),
            patch("arize._generated.api_client.TasksUpdateRequest"),
        ):
            result = tasks_client.update(task=_TASK_ID, name="y")

        assert result is expected

    def test_update_emits_alpha_prerelease_warning(
        self,
        tasks_client: TasksClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to update() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch("arize._generated.api_client.UpdateEvaluationTaskRequest"),
            patch("arize._generated.api_client.TasksUpdateRequest"),
        ):
            tasks_client.update(task=_TASK_ID, name="z")

        assert any(
            "ALPHA" in record.message and "tasks.update" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestTasksClientDelete:
    """Tests for TasksClient.delete()."""

    def test_delete_calls_api_with_task_id(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """delete() should resolve task and forward task_id to tasks_delete."""
        tasks_client.delete(task=_TASK_ID)

        mock_api.tasks_delete.assert_called_once_with(task_id=_TASK_ID)

    def test_delete_returns_none(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """delete() should return None when the API succeeds."""
        mock_api.tasks_delete.return_value = None

        result = tasks_client.delete(task=_TASK_ID)

        assert result is None

    def test_delete_emits_alpha_prerelease_warning(
        self,
        tasks_client: TasksClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to delete() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        tasks_client.delete(task=_TASK_ID)

        assert any(
            "ALPHA" in record.message and "tasks.delete" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestTasksClientTriggerRun:
    """Tests for TasksClient.trigger_run()."""

    def test_trigger_run_builds_request_and_calls_api(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """trigger_run() should build the inner request, wrap it, and call tasks_trigger_run."""
        with (
            patch(
                "arize._generated.api_client.TriggerEvaluationTaskRunRequest"
            ) as mock_inner_cls,
            patch(
                "arize._generated.api_client.TasksTriggerRunRequest"
            ) as mock_wrapper_cls,
        ):
            mock_inner = Mock()
            mock_inner_cls.return_value = mock_inner
            mock_wrapper = Mock()
            mock_wrapper_cls.return_value = mock_wrapper

            tasks_client.trigger_run(task=_TASK_ID)

        mock_inner_cls.assert_called_once_with(
            data_start_time=None,
            data_end_time=None,
            max_spans=None,
            override_evaluations=None,
            experiment_ids=None,
        )
        mock_wrapper_cls.assert_called_once_with(actual_instance=mock_inner)
        mock_api.tasks_trigger_run.assert_called_once_with(
            task_id=_TASK_ID,
            tasks_trigger_run_request=mock_wrapper,
        )

    def test_trigger_run_with_all_params(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """trigger_run() should forward all optional parameters."""
        from datetime import datetime, timezone

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)

        with (
            patch(
                "arize._generated.api_client.TriggerEvaluationTaskRunRequest"
            ) as mock_inner_cls,
            patch("arize._generated.api_client.TasksTriggerRunRequest"),
        ):
            mock_inner_cls.return_value = Mock()

            tasks_client.trigger_run(
                task=_TASK_ID,
                data_start_time=start,
                data_end_time=end,
                max_spans=500,
                override_evaluations=True,
                experiment_ids=["exp-1"],
            )

        _, kwargs = mock_inner_cls.call_args
        assert kwargs["data_start_time"] == start
        assert kwargs["data_end_time"] == end
        assert kwargs["max_spans"] == 500
        assert kwargs["override_evaluations"] is True
        assert kwargs["experiment_ids"] == ["exp-1"]

    def test_trigger_run_returns_api_response(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """trigger_run() should propagate the return value."""
        expected = Mock()
        mock_api.tasks_trigger_run.return_value = expected

        with (
            patch(
                "arize._generated.api_client.TriggerEvaluationTaskRunRequest"
            ),
            patch("arize._generated.api_client.TasksTriggerRunRequest"),
        ):
            result = tasks_client.trigger_run(task=_TASK_ID)

        assert result is expected

    def test_trigger_run_emits_alpha_prerelease_warning(
        self,
        tasks_client: TasksClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to trigger_run() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with (
            patch(
                "arize._generated.api_client.TriggerEvaluationTaskRunRequest"
            ),
            patch("arize._generated.api_client.TasksTriggerRunRequest"),
        ):
            tasks_client.trigger_run(task=_TASK_ID)

        assert any(
            "ALPHA" in record.message and "tasks.trigger_run" in record.message
            for record in caplog.records
        )

    @pytest.mark.parametrize(
        "field, value",
        [
            ("example_ids", ["ex-1"]),
            ("evaluation_task_ids", ["task-after-1"]),
        ],
    )
    def test_trigger_run_eval_rejects_run_experiment_only_fields(
        self,
        tasks_client: TasksClient,
        mock_api: Mock,
        field: str,
        value: object,
    ) -> None:
        """trigger_run() against an eval task should reject run_experiment-only fields."""
        mock_api.tasks_get.return_value.type = "template_evaluation"

        with pytest.raises(ValueError, match=field):
            tasks_client.trigger_run(task=_TASK_ID, **{field: value})


@pytest.mark.unit
class TestTasksClientTriggerRunExperiment:
    """Tests for TasksClient.trigger_run() against run_experiment tasks."""

    @staticmethod
    def _set_run_experiment_task(mock_api: Mock) -> None:
        mock_api.tasks_get.return_value.type = "run_experiment"

    def test_trigger_run_run_experiment_minimal(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """trigger_run() should build a TriggerRunExperimentTaskRunRequest with defaults."""
        self._set_run_experiment_task(mock_api)

        with (
            patch(
                "arize._generated.api_client.TriggerRunExperimentTaskRunRequest"
            ) as mock_inner_cls,
            patch(
                "arize._generated.api_client.TasksTriggerRunRequest"
            ) as mock_wrapper_cls,
        ):
            mock_inner = Mock()
            mock_inner_cls.return_value = mock_inner
            mock_wrapper = Mock()
            mock_wrapper_cls.return_value = mock_wrapper

            tasks_client.trigger_run(task=_TASK_ID, experiment_name="exp-name")

        mock_inner_cls.assert_called_once_with(
            experiment_name="exp-name",
            dataset_version_id=None,
            max_examples=None,
            example_ids=None,
            tracing_metadata=None,
            evaluation_task_ids=None,
        )
        mock_wrapper_cls.assert_called_once_with(actual_instance=mock_inner)
        mock_api.tasks_trigger_run.assert_called_once_with(
            task_id=_TASK_ID,
            tasks_trigger_run_request=mock_wrapper,
        )

    def test_trigger_run_run_experiment_forwards_all_fields(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """trigger_run() should forward every run_experiment-specific field."""
        self._set_run_experiment_task(mock_api)

        with (
            patch(
                "arize._generated.api_client.TriggerRunExperimentTaskRunRequest"
            ) as mock_inner_cls,
            patch("arize._generated.api_client.TasksTriggerRunRequest"),
        ):
            mock_inner_cls.return_value = Mock()

            tasks_client.trigger_run(
                task=_TASK_ID,
                experiment_name="exp-name",
                dataset_version_id="version-1",
                example_ids=["ex-1", "ex-2"],
                tracing_metadata={"env": "prod"},
                evaluation_task_ids=["task-after-1"],
            )

        _, kwargs = mock_inner_cls.call_args
        assert kwargs["experiment_name"] == "exp-name"
        assert kwargs["dataset_version_id"] == "version-1"
        assert kwargs["max_examples"] is None
        assert kwargs["example_ids"] == ["ex-1", "ex-2"]
        assert kwargs["tracing_metadata"] == {"env": "prod"}
        assert kwargs["evaluation_task_ids"] == ["task-after-1"]

    def test_trigger_run_run_experiment_with_max_examples(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """trigger_run() should accept max_examples on its own (no example_ids)."""
        self._set_run_experiment_task(mock_api)

        with (
            patch(
                "arize._generated.api_client.TriggerRunExperimentTaskRunRequest"
            ) as mock_inner_cls,
            patch("arize._generated.api_client.TasksTriggerRunRequest"),
        ):
            mock_inner_cls.return_value = Mock()

            tasks_client.trigger_run(
                task=_TASK_ID,
                experiment_name="exp-name",
                max_examples=50,
            )

        _, kwargs = mock_inner_cls.call_args
        assert kwargs["max_examples"] == 50
        assert kwargs["example_ids"] is None

    def test_trigger_run_run_experiment_requires_experiment_name(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """trigger_run() should raise when experiment_name is missing."""
        self._set_run_experiment_task(mock_api)

        with pytest.raises(ValueError, match="experiment_name"):
            tasks_client.trigger_run(task=_TASK_ID)

    def test_trigger_run_run_experiment_example_ids_and_max_examples_mutually_exclusive(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """trigger_run() should raise when both example_ids and max_examples are provided."""
        self._set_run_experiment_task(mock_api)

        with pytest.raises(ValueError, match="mutually exclusive"):
            tasks_client.trigger_run(
                task=_TASK_ID,
                experiment_name="exp-name",
                example_ids=["ex-1"],
                max_examples=10,
            )

    @pytest.mark.parametrize(
        "field, value",
        [
            ("max_spans", 500),
            ("override_evaluations", True),
            ("experiment_ids", ["exp-1"]),
        ],
    )
    def test_trigger_run_run_experiment_rejects_eval_only_fields(
        self,
        tasks_client: TasksClient,
        mock_api: Mock,
        field: str,
        value: object,
    ) -> None:
        """trigger_run() against a run_experiment task should reject eval-only fields."""
        self._set_run_experiment_task(mock_api)

        with pytest.raises(ValueError, match=field):
            tasks_client.trigger_run(
                task=_TASK_ID,
                experiment_name="exp-name",
                **{field: value},
            )


@pytest.mark.unit
class TestTasksClientListRuns:
    """Tests for TasksClient.list_runs()."""

    def test_list_runs_calls_api_with_all_params(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """list_runs() should forward all parameters to tasks_list_runs."""
        tasks_client.list_runs(
            task=_TASK_ID,
            status="completed",
            limit=25,
            cursor="cursor-abc",
        )

        mock_api.tasks_list_runs.assert_called_once_with(
            task_id=_TASK_ID,
            status="completed",
            limit=25,
            cursor="cursor-abc",
        )

    def test_list_runs_defaults(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """list_runs() should default status/cursor to None and limit to 100."""
        tasks_client.list_runs(task=_TASK_ID)

        mock_api.tasks_list_runs.assert_called_once_with(
            task_id=_TASK_ID,
            status=None,
            limit=100,
            cursor=None,
        )

    def test_list_runs_returns_api_response(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """list_runs() should propagate the return value."""
        expected = Mock()
        mock_api.tasks_list_runs.return_value = expected

        result = tasks_client.list_runs(task=_TASK_ID)

        assert result is expected

    def test_list_runs_emits_alpha_prerelease_warning(
        self,
        tasks_client: TasksClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        tasks_client.list_runs(task=_TASK_ID)

        assert any(
            "ALPHA" in record.message and "tasks.list_runs" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestTasksClientGetRun:
    """Tests for TasksClient.get_run()."""

    def test_get_run_calls_api_with_run_id(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """get_run() should pass run_id to task_runs_get."""
        tasks_client.get_run(run_id="run-456")

        mock_api.task_runs_get.assert_called_once_with(run_id="run-456")

    def test_get_run_returns_api_response(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """get_run() should propagate the return value."""
        expected = Mock()
        mock_api.task_runs_get.return_value = expected

        result = tasks_client.get_run(run_id="run-456")

        assert result is expected

    def test_get_run_emits_alpha_prerelease_warning(
        self,
        tasks_client: TasksClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        tasks_client.get_run(run_id="run-456")

        assert any(
            "ALPHA" in record.message and "tasks.get_run" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestTasksClientCancelRun:
    """Tests for TasksClient.cancel_run()."""

    def test_cancel_run_calls_api_with_run_id(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """cancel_run() should pass run_id to task_runs_cancel."""
        tasks_client.cancel_run(run_id="run-456")

        mock_api.task_runs_cancel.assert_called_once_with(run_id="run-456")

    def test_cancel_run_returns_api_response(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """cancel_run() should propagate the return value."""
        expected = Mock()
        mock_api.task_runs_cancel.return_value = expected

        result = tasks_client.cancel_run(run_id="run-456")

        assert result is expected

    def test_cancel_run_emits_alpha_prerelease_warning(
        self,
        tasks_client: TasksClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        tasks_client.cancel_run(run_id="run-456")

        assert any(
            "ALPHA" in record.message and "tasks.cancel_run" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestTasksClientWaitForRun:
    """Tests for TasksClient.wait_for_run()."""

    def _make_run(self, status: str) -> Mock:
        run = Mock()
        run.status = status
        return run

    def test_returns_immediately_on_terminal_status(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """wait_for_run() should return at once when the run is already terminal."""
        for status in ("completed", "failed", "cancelled"):
            mock_api.task_runs_get.return_value = self._make_run(status)
            result = tasks_client.wait_for_run(run_id="run-456")
            assert result.status == status

    def test_polls_until_terminal(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """wait_for_run() should keep polling until a terminal status appears."""
        runs = [
            self._make_run("pending"),
            self._make_run("running"),
            self._make_run("completed"),
        ]
        mock_api.task_runs_get.side_effect = runs

        with patch("time.sleep") as mock_sleep:
            result = tasks_client.wait_for_run(
                run_id="run-456", poll_interval=2.0
            )

        assert result.status == "completed"
        assert mock_api.task_runs_get.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(2.0)

    def test_raises_timeout_error(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """wait_for_run() should raise TimeoutError when timeout is exceeded."""
        mock_api.task_runs_get.return_value = self._make_run("running")

        with (
            patch("time.sleep"),
            patch(
                "time.monotonic",
                side_effect=[0.0, 0.0, 5.0, 5.0, 999.0],
            ),
            pytest.raises(TimeoutError, match="run-456"),
        ):
            tasks_client.wait_for_run(
                run_id="run-456", poll_interval=1.0, timeout=3.0
            )

    def test_default_poll_interval_and_timeout(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """wait_for_run() should use the module-level defaults when not specified."""
        mock_api.task_runs_get.return_value = self._make_run("completed")

        # Patch sleep so we don't actually sleep; confirm defaults by checking
        # the import-level constants.
        assert _DEFAULT_POLL_INTERVAL == 5.0
        assert _DEFAULT_TIMEOUT == 600.0

        with patch("time.sleep"):
            result = tasks_client.wait_for_run(run_id="run-456")

        assert result.status == "completed"

    def test_wait_for_run_emits_alpha_prerelease_warning(
        self,
        tasks_client: TasksClient,
        mock_api: Mock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)
        mock_api.task_runs_get.return_value = self._make_run("completed")

        tasks_client.wait_for_run(run_id="run-456")

        assert any(
            "ALPHA" in record.message and "tasks.wait_for_run" in record.message
            for record in caplog.records
        )

    def test_raises_value_error_on_zero_timeout(
        self, tasks_client: TasksClient
    ) -> None:
        """wait_for_run() should raise ValueError when timeout <= 0."""
        with pytest.raises(ValueError, match="timeout"):
            tasks_client.wait_for_run(run_id="run-456", timeout=0.0)

    def test_raises_value_error_on_negative_timeout(
        self, tasks_client: TasksClient
    ) -> None:
        """wait_for_run() should raise ValueError when timeout is negative."""
        with pytest.raises(ValueError, match="timeout"):
            tasks_client.wait_for_run(run_id="run-456", timeout=-1.0)

    def test_raises_value_error_on_zero_poll_interval(
        self, tasks_client: TasksClient
    ) -> None:
        """wait_for_run() should raise ValueError when poll_interval <= 0."""
        with pytest.raises(ValueError, match="poll_interval"):
            tasks_client.wait_for_run(run_id="run-456", poll_interval=0.0)
