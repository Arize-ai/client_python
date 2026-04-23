"""Client implementation for managing tasks and task runs in the Arize platform."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Final, Literal

from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import (
    _find_dataset_id,
    _find_project_id,
    _find_task_id,
    _resolve_resource,
)

if TYPE_CHECKING:
    import builtins
    from datetime import datetime

    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration
    from arize.tasks.types import (
        Task,
        TaskRun,
        TasksCreateRequestEvaluatorsInner,
        TasksList200Response,
        TasksListRuns200Response,
    )

# Shared type aliases — kept here so method signatures and _TERMINAL_STATUSES
# stay in sync with a single source of truth.
TaskType = Literal["template_evaluation", "code_evaluation"]
RunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_DEFAULT_POLL_INTERVAL = 5.0  # seconds
_DEFAULT_TIMEOUT = 600.0  # seconds


# Sentinel for TasksClient.update — omit field from PATCH body vs explicit values.
# Defined as a class (rather than `object()`) so method signatures can spell
# out `str | _Missing` instead of the looser `str | object`.
class _Missing:
    """Sentinel type used to distinguish "omitted" from explicit ``None``."""


_MISSING: Final[_Missing] = _Missing()


class TasksClient:
    """Client for managing Arize tasks and task runs.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The tasks client is a thin wrapper around the generated REST API client,
    using the shared generated API client owned by
    :class:`arize.config.SDKConfiguration`.
    """

    def __init__(
        self, *, sdk_config: SDKConfiguration, generated_client: ApiClient
    ) -> None:
        """
        Args:
            sdk_config: Resolved SDK configuration.
            generated_client: Shared generated API client instance.
        """  # noqa: D205, D212
        self._sdk_config = sdk_config

        # Import at runtime so it's still lazy and extras-gated by the parent
        from arize._generated import api_client as gen

        self._api = gen.TasksApi(generated_client)
        self._projects_api = gen.ProjectsApi(generated_client)
        self._datasets_api = gen.DatasetsApi(generated_client)

    # -------------------------------------------------------------------------
    # Tasks
    # -------------------------------------------------------------------------

    @prerelease_endpoint(key="tasks.list", stage=ReleaseStage.ALPHA)
    def list(
        self,
        *,
        name: str | None = None,
        project: str | None = None,
        dataset: str | None = None,
        space: str | None = None,
        task_type: TaskType | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> TasksList200Response:
        """List tasks the user has access to.

        Results support cursor-based pagination. Optionally filter by space,
        project, dataset, or task type.

        Args:
            name: Optional case-insensitive substring filter on the task name.
            project: Optional project name or global ID (base64) to filter
                results. If the value is a name, ``space`` must also be
                provided.
            dataset: Optional dataset name or global ID (base64) to filter
                results. If the value is a name, ``space`` must also be
                provided.
            space: Optional space name or ID used to disambiguate name-based
                resolution for ``project`` and ``dataset``. If the value is a
                base64-encoded resource ID it is treated as a space ID;
                otherwise it is used as a case-insensitive substring filter on
                the space name.
            task_type: Optional task type filter. One of
                ``"template_evaluation"`` or ``"code_evaluation"``.
            limit: Maximum number of tasks to return (1-100).
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated task list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails.
        """
        project_id = (
            _find_project_id(
                api=self._projects_api,
                project=project,
                space=space,
            )
            if project
            else None
        )
        dataset_id = (
            _find_dataset_id(
                api=self._datasets_api,
                dataset=dataset,
                space=space,
            )
            if dataset
            else None
        )
        resolved_space = _resolve_resource(space)
        return self._api.tasks_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=name,
            project_id=project_id,
            dataset_id=dataset_id,
            type=task_type,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="tasks.get", stage=ReleaseStage.ALPHA)
    def get(self, *, task: str, space: str | None = None) -> Task:
        """Get a task by name or ID.

        Args:
            task: Task name or global ID (base64). If the value looks like an
                ID it is used directly; otherwise it is resolved by name.
            space: Optional space name or ID used to disambiguate the task
                lookup. Recommended when resolving by name.

        Returns:
            The task with its full configuration.

        Raises:
            ApiException: If the API request fails
                (for example, task not found).
        """
        task_id = _find_task_id(
            api=self._api,
            task=task,
            space=space,
        )
        return self._api.tasks_get(task_id=task_id)

    @prerelease_endpoint(key="tasks.create", stage=ReleaseStage.ALPHA)
    def create(
        self,
        *,
        name: str,
        task_type: TaskType,
        evaluators: builtins.list[TasksCreateRequestEvaluatorsInner],
        project: str | None = None,
        dataset: str | None = None,
        space: str | None = None,
        experiment_ids: builtins.list[str] | None = None,
        sampling_rate: float | None = None,
        is_continuous: bool | None = None,
        query_filter: str | None = None,
    ) -> Task:
        """Create a new evaluation task.

        Either ``project`` or ``dataset`` must be provided, but not both.
        When ``dataset`` is provided, at least one ``experiment_ids`` entry
        is required.

        Args:
            name: Task name (must be unique within the space).
            task_type: Task type. One of ``"template_evaluation"`` or
                ``"code_evaluation"``.
            evaluators: List of evaluators to attach. At least one is required.
                Each evaluator is a
                :class:`arize.tasks.types.TasksCreateRequestEvaluatorsInner`
                with the following fields:

                - ``evaluator_id`` — Evaluator global ID (base64). Required.
                - ``query_filter`` — Per-evaluator filter (AND-ed with
                  task-level filter). Optional.
                - ``column_mappings`` — Maps template variable names to column
                  names. Optional.

            project: Project name or global ID (base64). Required when
                ``dataset`` is not provided.
            dataset: Dataset name or global ID (base64). Required when
                ``project`` is not provided.
            space: Optional space name or ID used to disambiguate name-based
                resolution for ``project`` and ``dataset``.
            experiment_ids: Experiment global IDs (base64). Required (at least
                one) when ``dataset`` is provided. Must be omitted or empty
                for project-based tasks.
            sampling_rate: Fraction of data to evaluate (0-1). Only valid for
                project-based tasks.
            is_continuous: Whether to run the task continuously. Must be
                ``True`` or ``False`` for project-based tasks; must be
                ``False`` or omitted for dataset-based tasks.
            query_filter: Task-level query filter applied to all evaluators.

        Returns:
            The newly created task.

        Raises:
            ApiException: If the API request fails
                (for example, invalid payload or name conflict).
        """
        project_id = (
            _find_project_id(
                api=self._projects_api,
                project=project,
                space=space,
            )
            if project
            else None
        )
        dataset_id = (
            _find_dataset_id(
                api=self._datasets_api,
                dataset=dataset,
                space=space,
            )
            if dataset
            else None
        )

        from arize._generated import api_client as gen

        body = gen.TasksCreateRequest(
            name=name,
            type=task_type,
            evaluators=evaluators,
            project_id=project_id,
            dataset_id=dataset_id,
            experiment_ids=experiment_ids,
            sampling_rate=sampling_rate,
            is_continuous=is_continuous,
            query_filter=query_filter,
        )
        return self._api.tasks_create(tasks_create_request=body)

    @prerelease_endpoint(key="tasks.update", stage=ReleaseStage.ALPHA)
    def update(
        self,
        *,
        task: str,
        space: str | None = None,
        name: str | _Missing = _MISSING,
        sampling_rate: float | _Missing = _MISSING,
        is_continuous: bool | _Missing = _MISSING,
        query_filter: str | None | _Missing = _MISSING,
        evaluators: builtins.list[TasksCreateRequestEvaluatorsInner]
        | _Missing = _MISSING,
    ) -> Task:
        """Update mutable fields on an existing task.

        At least one mutable field must be provided. Pass ``None`` to
        ``query_filter`` to clear the existing filter; omit the argument to
        leave it unchanged.

        Args:
            task: Task name or global ID (base64). Names are resolved within
                the space when ``space`` is provided.
            space: Optional space name or ID used to disambiguate task name
                resolution.
            name: New display name for the task.
            sampling_rate: Fraction of data to evaluate (0-1). Project-based
                tasks only.
            is_continuous: Whether the task runs continuously.
            query_filter: Task-level query filter, or ``None`` to clear the
                filter.
            evaluators: Full replacement list of evaluators (at least one when
                provided).

        Returns:
            The updated task.

        Raises:
            ValueError: If no update fields were provided.
            ApiException: If the API request fails.
        """
        payload: dict[str, Any] = {}
        if name is not _MISSING:
            payload["name"] = name
        if sampling_rate is not _MISSING:
            payload["sampling_rate"] = sampling_rate
        if is_continuous is not _MISSING:
            payload["is_continuous"] = is_continuous
        if query_filter is not _MISSING:
            payload["query_filter"] = query_filter
        if evaluators is not _MISSING:
            payload["evaluators"] = evaluators

        if not payload:
            raise ValueError(
                "At least one update field must be provided "
                "(name, sampling_rate, is_continuous, query_filter, or evaluators).",
            )

        from arize._generated import api_client as gen

        task_id = _find_task_id(
            api=self._api,
            task=task,
            space=space,
        )
        body = gen.TasksUpdateRequest(**payload)
        return self._api.tasks_update(
            task_id=task_id,
            tasks_update_request=body,
        )

    @prerelease_endpoint(key="tasks.delete", stage=ReleaseStage.ALPHA)
    def delete(self, *, task: str, space: str | None = None) -> None:
        """Delete a task and its associated configuration.

        Args:
            task: Task name or global ID (base64).
            space: Optional space name or ID used when resolving by task name.

        Raises:
            ApiException: If the API request fails.
        """
        task_id = _find_task_id(
            api=self._api,
            task=task,
            space=space,
        )
        self._api.tasks_delete(task_id=task_id)

    # -------------------------------------------------------------------------
    # Task runs
    # -------------------------------------------------------------------------

    @prerelease_endpoint(key="tasks.trigger_run", stage=ReleaseStage.ALPHA)
    def trigger_run(
        self,
        *,
        task: str,
        space: str | None = None,
        data_start_time: datetime | None = None,
        data_end_time: datetime | None = None,
        max_spans: int | None = None,
        override_evaluations: bool | None = None,
        experiment_ids: builtins.list[str] | None = None,
    ) -> TaskRun:
        """Trigger an on-demand run for a task.

        Args:
            task: Task name or global ID (base64) to trigger a run for.
            space: Optional space name or ID used to disambiguate the task
                lookup. Recommended when resolving by name.
            data_start_time: Optional ISO 8601 start of the data window to
                evaluate.
            data_end_time: Optional ISO 8601 end of the data window to
                evaluate. Defaults to now when omitted.
            max_spans: Maximum number of spans to process (default 10 000).
            override_evaluations: Whether to re-evaluate data that already
                has evaluation labels. Defaults to ``False``.
            experiment_ids: Experiment global IDs (base64) to run against.
                Only applicable for dataset-based tasks.

        Returns:
            The newly created task run (initially in ``"pending"`` status).

        Raises:
            ApiException: If the API request fails.
        """
        task_id = _find_task_id(
            api=self._api,
            task=task,
            space=space,
        )

        from arize._generated import api_client as gen

        body = gen.TasksTriggerRunRequest(
            data_start_time=data_start_time,
            data_end_time=data_end_time,
            max_spans=max_spans,
            override_evaluations=override_evaluations,
            experiment_ids=experiment_ids,
        )
        return self._api.tasks_trigger_run(
            task_id=task_id,
            tasks_trigger_run_request=body,
        )

    @prerelease_endpoint(key="tasks.list_runs", stage=ReleaseStage.ALPHA)
    def list_runs(
        self,
        *,
        task: str,
        space: str | None = None,
        status: RunStatus | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> TasksListRuns200Response:
        """List runs for a task.

        Results support cursor-based pagination. Optionally filter by run
        status.

        Args:
            task: Task name or global ID (base64) to list runs for.
            space: Optional space name or ID used to disambiguate the task
                lookup. Recommended when resolving by name.
            status: Optional run status filter. One of ``"pending"``,
                ``"running"``, ``"completed"``, ``"failed"``, or
                ``"cancelled"``.
            limit: Maximum number of runs to return (1-100).
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated task run list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails.
        """
        task_id = _find_task_id(
            api=self._api,
            task=task,
            space=space,
        )
        return self._api.tasks_list_runs(
            task_id=task_id,
            status=status,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="tasks.get_run", stage=ReleaseStage.ALPHA)
    def get_run(self, *, run_id: str) -> TaskRun:
        """Get a task run by its global ID.

        Args:
            run_id: Task run global ID (base64) to retrieve.

        Returns:
            The task run with its current status and statistics.

        Raises:
            ApiException: If the API request fails
                (for example, run not found).
        """
        return self._api.task_runs_get(run_id=run_id)

    @prerelease_endpoint(key="tasks.cancel_run", stage=ReleaseStage.ALPHA)
    def cancel_run(self, *, run_id: str) -> TaskRun:
        """Cancel a task run.

        Only valid when the run's current status is ``"pending"`` or
        ``"running"``.

        Args:
            run_id: Task run global ID (base64) to cancel.

        Returns:
            The updated task run with status ``"cancelled"``.

        Raises:
            ApiException: If the API request fails
                (for example, run not found or already in terminal state).
        """
        return self._api.task_runs_cancel(run_id=run_id)

    @prerelease_endpoint(key="tasks.wait_for_run", stage=ReleaseStage.ALPHA)
    def wait_for_run(
        self,
        *,
        run_id: str,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> TaskRun:
        """Poll a task run until it reaches a terminal state.

        Repeatedly calls :meth:`get_run` at ``poll_interval``-second intervals
        until the run's status is one of ``"completed"``, ``"failed"``, or
        ``"cancelled"``, or until ``timeout`` seconds have elapsed.

        Args:
            run_id: Task run global ID (base64) to wait for.
            poll_interval: Seconds between polling attempts. Defaults to 5.
            timeout: Maximum seconds to wait before raising
                ``TimeoutError``. Defaults to 600.

        Returns:
            The task run in its terminal state.

        Raises:
            ValueError: If ``timeout`` or ``poll_interval`` is not positive.
            TimeoutError: If the run does not reach a terminal state within
                ``timeout`` seconds.
            ApiException: If any polling request fails.
        """
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout!r}")
        if poll_interval <= 0:
            raise ValueError(
                f"poll_interval must be positive, got {poll_interval!r}"
            )
        deadline = time.monotonic() + timeout
        while True:
            # Call the generated API directly instead of self.get_run() to
            # avoid firing the @prerelease_endpoint warning on every iteration;
            # the outer wait_for_run() method is already decorated.
            run = self._api.task_runs_get(run_id=run_id)
            if run.status in _TERMINAL_STATUSES:
                return run
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Task run {run_id!r} did not reach a terminal state "
                    f"within {timeout:.0f}s (last status: {run.status!r})."
                )
            sleep_time = min(poll_interval, remaining)
            logger.debug(
                "Task run %r status=%r; polling again in %.1fs",
                run_id,
                run.status,
                sleep_time,
            )
            time.sleep(sleep_time)
