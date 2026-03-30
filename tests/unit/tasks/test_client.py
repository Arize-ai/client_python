"""Unit tests for src/arize/tasks/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.tasks.client import (
    _DEFAULT_POLL_INTERVAL,
    _DEFAULT_TIMEOUT,
    TasksClient,
)

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

    def test_create_builds_request_and_calls_api(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create() should build the request object and call tasks_create."""
        mock_evaluator = Mock()

        with patch(
            "arize._generated.api_client.TasksCreateRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            tasks_client.create(
                name="my-task",
                task_type="template_evaluation",
                evaluators=[mock_evaluator],
                project=_PROJECT_ID,
            )

        mock_request_cls.assert_called_once_with(
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
        mock_api.tasks_create.assert_called_once_with(
            tasks_create_request=mock_body
        )

    def test_create_with_dataset_and_experiments(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create() should forward dataset and experiment_ids."""
        with patch(
            "arize._generated.api_client.TasksCreateRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            tasks_client.create(
                name="ds-task",
                task_type="code_evaluation",
                evaluators=[Mock()],
                dataset=_DATASET_ID,
                experiment_ids=["exp-1", "exp-2"],
            )

        _, kwargs = mock_request_cls.call_args
        assert kwargs["dataset_id"] == _DATASET_ID
        assert kwargs["experiment_ids"] == ["exp-1", "exp-2"]
        assert kwargs["project_id"] is None

    def test_create_with_optional_fields(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create() should forward optional sampling_rate, is_continuous, query_filter."""
        with patch(
            "arize._generated.api_client.TasksCreateRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            tasks_client.create(
                name="my-task",
                task_type="template_evaluation",
                evaluators=[Mock()],
                project=_PROJECT_ID,
                sampling_rate=0.5,
                is_continuous=True,
                query_filter="span_kind == 'LLM'",
            )

        _, kwargs = mock_request_cls.call_args
        assert kwargs["sampling_rate"] == 0.5
        assert kwargs["is_continuous"] is True
        assert kwargs["query_filter"] == "span_kind == 'LLM'"

    def test_create_returns_api_response(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from tasks_create."""
        expected = Mock()
        mock_api.tasks_create.return_value = expected

        with patch("arize._generated.api_client.TasksCreateRequest"):
            result = tasks_client.create(
                name="my-task",
                task_type="template_evaluation",
                evaluators=[Mock()],
                project=_PROJECT_ID,
            )

        assert result is expected

    def test_create_emits_alpha_prerelease_warning(
        self,
        tasks_client: TasksClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call to create() should emit the ALPHA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        with patch("arize._generated.api_client.TasksCreateRequest"):
            tasks_client.create(
                name="my-task",
                task_type="template_evaluation",
                evaluators=[Mock()],
                project=_PROJECT_ID,
            )

        assert any(
            "ALPHA" in record.message and "tasks.create" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestTasksClientTriggerRun:
    """Tests for TasksClient.trigger_run()."""

    def test_trigger_run_builds_request_and_calls_api(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """trigger_run() should build the request and call tasks_trigger_run."""
        with patch(
            "arize._generated.api_client.TasksTriggerRunRequest"
        ) as mock_request_cls:
            mock_body = Mock()
            mock_request_cls.return_value = mock_body

            tasks_client.trigger_run(task=_TASK_ID)

        mock_request_cls.assert_called_once_with(
            data_start_time=None,
            data_end_time=None,
            max_spans=None,
            override_evaluations=None,
            experiment_ids=None,
        )
        mock_api.tasks_trigger_run.assert_called_once_with(
            task_id=_TASK_ID,
            tasks_trigger_run_request=mock_body,
        )

    def test_trigger_run_with_all_params(
        self, tasks_client: TasksClient, mock_api: Mock
    ) -> None:
        """trigger_run() should forward all optional parameters."""
        from datetime import datetime, timezone

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)

        with patch(
            "arize._generated.api_client.TasksTriggerRunRequest"
        ) as mock_request_cls:
            mock_request_cls.return_value = Mock()

            tasks_client.trigger_run(
                task=_TASK_ID,
                data_start_time=start,
                data_end_time=end,
                max_spans=500,
                override_evaluations=True,
                experiment_ids=["exp-1"],
            )

        _, kwargs = mock_request_cls.call_args
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

        with patch("arize._generated.api_client.TasksTriggerRunRequest"):
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

        with patch("arize._generated.api_client.TasksTriggerRunRequest"):
            tasks_client.trigger_run(task=_TASK_ID)

        assert any(
            "ALPHA" in record.message and "tasks.trigger_run" in record.message
            for record in caplog.records
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
