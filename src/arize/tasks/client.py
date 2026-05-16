"""Client implementation for managing tasks and task runs in the Arize platform."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Final

from arize._generated.api_client.models.run_configuration import (
    RunConfiguration,
)
from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.tasks.types import (
    LlmGenerationRunConfig,
    Task,
    TasksList200Response,
    TaskType,
    TemplateEvaluationRunConfig,
)
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
        BaseEvaluationTaskRequestEvaluatorsInner,
        RunStatus,
        TaskRun,
        TasksListRuns200Response,
    )

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_run_configuration(
        item: RunConfiguration
        | LlmGenerationRunConfig
        | TemplateEvaluationRunConfig
        | dict,
    ) -> RunConfiguration:
        """Normalize a run configuration to a properly wrapped ``RunConfiguration``.

        Accepts:
        - An already-wrapped ``RunConfiguration`` (returned as-is).
        - An unwrapped inner type (``LlmGenerationRunConfig`` or
          ``TemplateEvaluationRunConfig``), which is wrapped automatically.
        - A plain ``dict`` whose keys match one of the inner schemas; parsed via
          ``RunConfiguration.from_dict``.
        """
        if isinstance(item, RunConfiguration):
            return item
        if isinstance(
            item, (LlmGenerationRunConfig, TemplateEvaluationRunConfig)
        ):
            return RunConfiguration(item)
        if isinstance(item, dict):
            return RunConfiguration.from_dict(item)
        raise TypeError(
            f"run_configuration must be RunConfiguration, LlmGenerationRunConfig, "
            f"TemplateEvaluationRunConfig, or dict; got {type(item)!r}"
        )

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
        result = self._api.tasks_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=name,
            project_id=project_id,
            dataset_id=dataset_id,
            type=task_type,
            limit=limit,
            cursor=cursor,
        )
        return TasksList200Response.model_validate(result, from_attributes=True)

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
        result = self._api.tasks_get(task_id=task_id)
        return Task.model_validate(result, from_attributes=True)

    def _create(
        self,
        *,
        name: str,
        task_type: TaskType,
        evaluators: builtins.list[BaseEvaluationTaskRequestEvaluatorsInner]
        | None = None,
        run_configuration: RunConfiguration
        | LlmGenerationRunConfig
        | TemplateEvaluationRunConfig
        | dict
        | None = None,
        project: str | None = None,
        dataset: str | None = None,
        space: str | None = None,
        experiment_ids: builtins.list[str] | None = None,
        sampling_rate: float | None = None,
        is_continuous: bool | None = None,
        query_filter: str | None = None,
    ) -> Task:
        """Create a new task.

        The required arguments depend on ``task_type``:

        - ``"template_evaluation"`` / ``"code_evaluation"``: ``evaluators``
          is required. Either ``project`` or ``dataset`` must be provided.
          When ``dataset`` is provided, at least one entry in
          ``experiment_ids`` is required.
        - ``"run_experiment"``: ``run_configuration`` and ``dataset`` are
          required. Eval-only fields (``evaluators``, ``project``,
          ``sampling_rate``, ``is_continuous``, ``query_filter``,
          ``experiment_ids``) must be omitted.

        Args:
            name: Task name (must be unique within the space).
            task_type: Task type: ``"template_evaluation"``,
                ``"code_evaluation"``, or ``"run_experiment"``.
            evaluators: List of evaluators to attach. Required for eval task
                types; must be omitted for ``"run_experiment"``. Each entry is
                a :class:`arize.tasks.types.BaseEvaluationTaskRequestEvaluatorsInner`
                with the following fields:

                - ``evaluator_id`` — Evaluator global ID (base64). Required.
                - ``query_filter`` — Per-evaluator filter (AND-ed with
                  task-level filter). Optional.
                - ``column_mappings`` — Maps template variable names to column
                  names. Optional.

            run_configuration: Experiment run configuration. Required for
                ``"run_experiment"`` tasks; must be omitted for eval task
                types. Use
                :class:`arize.tasks.types.LlmGenerationRunConfig` or
                :class:`arize.tasks.types.TemplateEvaluationRunConfig`
                wrapped in
                :class:`arize.tasks.types.RunConfiguration`.
            project: Project name or global ID (base64). For eval tasks,
                required when ``dataset`` is not provided.
            dataset: Dataset name or global ID (base64). Required for
                ``"run_experiment"`` tasks; for eval tasks, required when
                ``project`` is not provided.
            space: Optional space name or ID used to disambiguate name-based
                resolution for ``project`` and ``dataset``.
            experiment_ids: Experiment global IDs (base64). For eval tasks:
                required (at least one) when ``dataset`` is provided; must
                be omitted for project-based tasks. Not applicable for
                ``"run_experiment"`` tasks.
            sampling_rate: Fraction of data to evaluate (0-1). Only valid for
                project-based eval tasks.
            is_continuous: Whether to run the task continuously. Only valid
                for eval tasks.
            query_filter: Task-level query filter applied to all evaluators.
                Only valid for eval tasks.

        Returns:
            The newly created task.

        Raises:
            ValueError: If required fields are missing or mutually exclusive
                fields are combined.
            ApiException: If the API request fails
                (for example, invalid payload or name conflict).
        """
        from arize._generated import api_client as gen

        if task_type == "run_experiment":
            eval_only = {
                k: v
                for k, v in {
                    "evaluators": evaluators,
                    "project": project,
                    "sampling_rate": sampling_rate,
                    "is_continuous": is_continuous,
                    "query_filter": query_filter,
                    "experiment_ids": experiment_ids,
                }.items()
                if v is not None
            }
            if eval_only:
                raise ValueError(
                    f"run_experiment tasks do not support eval-only fields: "
                    f"{', '.join(eval_only)}. Use 'run_configuration' and 'dataset' instead.",
                )
            if run_configuration is None:
                raise ValueError(
                    "'run_configuration' is required for run_experiment tasks."
                )
            if not dataset:
                raise ValueError(
                    "'dataset' is required for run_experiment tasks."
                )
            run_exp_dataset_id = _find_dataset_id(
                api=self._datasets_api,
                dataset=dataset,
                space=space,
            )
            run_exp_inner = gen.CreateRunExperimentTaskRequest(
                name=name,
                type="run_experiment",
                dataset_id=run_exp_dataset_id,
                run_configuration=self._coerce_run_configuration(
                    run_configuration
                ),
            )
            body = gen.TasksCreateRequest(actual_instance=run_exp_inner)
            result = self._api.tasks_create(tasks_create_request=body)
            return Task.model_validate(result, from_attributes=True)
        if run_configuration is not None:
            raise ValueError(
                f"'run_configuration' is only valid for run_experiment tasks, "
                f"not '{task_type}'.",
            )
        if evaluators is None:
            raise ValueError(
                f"'evaluators' is required for '{task_type}' tasks."
            )
        project_id = (
            _find_project_id(
                api=self._projects_api,
                project=project,
                space=space,
            )
            if project
            else None
        )
        dataset_id: str | None = (
            _find_dataset_id(
                api=self._datasets_api,
                dataset=dataset,
                space=space,
            )
            if dataset
            else None
        )
        inner_cls = (
            gen.CreateTemplateEvaluationTaskRequest
            if task_type == "template_evaluation"
            else gen.CreateCodeEvaluationTaskRequest
        )
        eval_inner = inner_cls(
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
        body = gen.TasksCreateRequest(actual_instance=eval_inner)
        result = self._api.tasks_create(tasks_create_request=body)
        return Task.model_validate(result, from_attributes=True)

    @prerelease_endpoint(
        key="tasks.create_evaluation_task", stage=ReleaseStage.ALPHA
    )
    def create_evaluation_task(
        self,
        *,
        name: str,
        task_type: TaskType,
        evaluators: builtins.list[BaseEvaluationTaskRequestEvaluatorsInner],
        project: str | None = None,
        dataset: str | None = None,
        space: str | None = None,
        experiment_ids: builtins.list[str] | None = None,
        sampling_rate: float | None = None,
        is_continuous: bool | None = None,
        query_filter: str | None = None,
    ) -> Task:
        """Create a new evaluation task.

        A typed convenience wrapper around the internal task-creation logic for
        ``"template_evaluation"`` and ``"code_evaluation"`` task types.
        Prefer this method when creating evaluation tasks
        for a cleaner, narrowly-typed signature.

        Args:
            name: Task name (must be unique within the space).
            task_type: Task type: ``"template_evaluation"`` or
                ``"code_evaluation"``.
            evaluators: List of evaluators to attach (at least one required).
                Each entry is a
                :class:`arize.tasks.types.BaseEvaluationTaskRequestEvaluatorsInner`
                with the following fields:

                - ``evaluator_id`` — Evaluator global ID (base64). Required.
                - ``query_filter`` — Per-evaluator filter. Optional.
                - ``column_mappings`` — Template variable name mappings. Optional.

            project: Project name or global ID (base64). Required when
                ``dataset`` is not provided.
            dataset: Dataset name or global ID (base64). Required when
                ``project`` is not provided.
            space: Optional space name or ID used to disambiguate name-based
                resolution for ``project`` and ``dataset``.
            experiment_ids: Experiment global IDs (base64). Required (at least
                one) when ``dataset`` is provided.
            sampling_rate: Fraction of data to evaluate (0-1). Only valid for
                project-based tasks.
            is_continuous: Whether to run the task continuously. Only valid
                for project-based tasks.
            query_filter: Task-level query filter applied to all evaluators.

        Returns:
            The newly created task.

        Raises:
            ValueError: If required fields are missing or mutually exclusive
                fields are combined.
            ApiException: If the API request fails.
        """
        return self._create(
            name=name,
            task_type=task_type,
            evaluators=evaluators,
            project=project,
            dataset=dataset,
            space=space,
            experiment_ids=experiment_ids,
            sampling_rate=sampling_rate,
            is_continuous=is_continuous,
            query_filter=query_filter,
        )

    @prerelease_endpoint(
        key="tasks.create_run_experiment_task", stage=ReleaseStage.ALPHA
    )
    def create_run_experiment_task(
        self,
        *,
        name: str,
        dataset: str,
        run_configuration: RunConfiguration
        | LlmGenerationRunConfig
        | TemplateEvaluationRunConfig
        | dict,
        space: str | None = None,
    ) -> Task:
        """Create a new ``run_experiment`` task.

        A typed convenience wrapper around the internal task-creation logic for
        ``"run_experiment"`` task types. The server drives all LLM calls
        using the AI integration specified in ``run_configuration`` — no
        local callable is required.

        To create and immediately trigger a run in one call, use
        ``create_and_run_experiment_task`` (available separately).

        Args:
            name: Task name (must be unique within the space).
            dataset: Dataset name or global ID (base64) to run the
                experiment against.
            run_configuration: Discriminated experiment configuration. Use
                :class:`arize.tasks.types.LlmGenerationRunConfig` or
                :class:`arize.tasks.types.TemplateEvaluationRunConfig`
                wrapped in :class:`arize.tasks.types.RunConfiguration`.
            space: Optional space name or ID used to resolve ``dataset``
                by name.

        Returns:
            The newly created task.

        Raises:
            ApiException: If the API request fails.
        """
        return self._create(
            name=name,
            task_type=TaskType.RUN_EXPERIMENT,
            run_configuration=run_configuration,
            dataset=dataset,
            space=space,
        )

    @prerelease_endpoint(key="tasks.update", stage=ReleaseStage.ALPHA)
    def update(
        self,
        *,
        task: str,
        space: str | None = None,
        name: str | _Missing = _MISSING,
        # Evaluation-task fields
        sampling_rate: float | _Missing = _MISSING,
        is_continuous: bool | _Missing = _MISSING,
        query_filter: str | None | _Missing = _MISSING,
        evaluators: builtins.list[BaseEvaluationTaskRequestEvaluatorsInner]
        | _Missing = _MISSING,
        # run_experiment-task fields
        run_configuration: RunConfiguration
        | LlmGenerationRunConfig
        | TemplateEvaluationRunConfig
        | dict
        | _Missing = _MISSING,
    ) -> Task:
        """Update mutable fields on an existing task.

        Dispatches based on the task's type — resolves the task by ID or name
        first, then GETs it to determine whether it is an evaluation task or a
        ``run_experiment`` task, and builds the appropriate PATCH body.

        At least one mutable field must be provided. Pass ``None`` to
        ``query_filter`` to clear the existing filter; omit the argument to
        leave it unchanged.

        For **evaluation tasks** (``template_evaluation`` /
        ``code_evaluation``):

        - Valid fields: ``name``, ``sampling_rate``, ``is_continuous``,
          ``query_filter``, ``evaluators``.
        - ``run_configuration`` must not be provided.

        For **run_experiment tasks**:

        - Valid fields: ``name``, ``run_configuration``.
        - Evaluation-only fields (``sampling_rate``, ``is_continuous``,
          ``query_filter``, ``evaluators``) must not be provided.

        Args:
            task: Task name or global ID (base64). Names are resolved within
                the space when ``space`` is provided.
            space: Optional space name or ID used to disambiguate task name
                resolution.
            name: New display name for the task.
            sampling_rate: Fraction of data to evaluate (0-1). Evaluation
                tasks only, project-based tasks only.
            is_continuous: Whether the task runs continuously. Evaluation
                tasks only.
            query_filter: Task-level query filter, or ``None`` to clear the
                filter. Evaluation tasks only.
            evaluators: Full replacement list of evaluators (at least one when
                provided). Evaluation tasks only.
            run_configuration: Replacement run configuration. When provided
                the entire stored config is atomically replaced.
                ``run_experiment`` tasks only.

        Returns:
            The updated task.

        Raises:
            ValueError: If no update fields were provided, or if a field is
                not valid for the resolved task type.
            ApiException: If the API request fails.
        """
        from arize._generated import api_client as gen

        task_id = _find_task_id(
            api=self._api,
            task=task,
            space=space,
        )
        task_obj = self._api.tasks_get(task_id=task_id)

        if task_obj.type == "run_experiment":
            # Validate that no eval-only fields were supplied.
            eval_only_supplied = {
                k: v
                for k, v in {
                    "sampling_rate": sampling_rate,
                    "is_continuous": is_continuous,
                    "query_filter": query_filter,
                    "evaluators": evaluators,
                }.items()
                if not isinstance(v, _Missing)
            }
            if eval_only_supplied:
                raise ValueError(
                    "Fields not valid for run_experiment tasks: "
                    f"{', '.join(eval_only_supplied)}. "
                    "Only 'name' and 'run_configuration' may be updated.",
                )
            run_exp_payload: dict[str, Any] = {}
            if not isinstance(name, _Missing):
                run_exp_payload["name"] = name
            if not isinstance(run_configuration, _Missing):
                run_exp_payload["run_configuration"] = (
                    self._coerce_run_configuration(run_configuration)
                )
            if not run_exp_payload:
                raise ValueError(
                    "At least one update field must be provided for "
                    "run_experiment tasks (name or run_configuration).",
                )
            inner_run_exp = gen.UpdateRunExperimentTaskRequest(
                **run_exp_payload
            )
            body = gen.TasksUpdateRequest(actual_instance=inner_run_exp)
        else:
            # Evaluation task.
            if not isinstance(run_configuration, _Missing):
                raise ValueError(
                    "'run_configuration' is only valid for run_experiment tasks, "
                    f"not '{task_obj.type}'.",
                )
            eval_payload: dict[str, Any] = {}
            if not isinstance(name, _Missing):
                eval_payload["name"] = name
            if not isinstance(sampling_rate, _Missing):
                eval_payload["sampling_rate"] = sampling_rate
            if not isinstance(is_continuous, _Missing):
                eval_payload["is_continuous"] = is_continuous
            if not isinstance(query_filter, _Missing):
                eval_payload["query_filter"] = query_filter
            if not isinstance(evaluators, _Missing):
                eval_payload["evaluators"] = evaluators
            if not eval_payload:
                raise ValueError(
                    "At least one update field must be provided "
                    "(name, sampling_rate, is_continuous, query_filter, or evaluators).",
                )
            inner = gen.UpdateEvaluationTaskRequest(**eval_payload)
            body = gen.TasksUpdateRequest(actual_instance=inner)

        result = self._api.tasks_update(
            task_id=task_id,
            tasks_update_request=body,
        )
        return Task.model_validate(result, from_attributes=True)

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
        # Evaluation-task fields
        data_start_time: datetime | None = None,
        data_end_time: datetime | None = None,
        max_spans: int | None = None,
        override_evaluations: bool | None = None,
        experiment_ids: builtins.list[str] | None = None,
        # run_experiment-task fields
        experiment_name: str | None = None,
        dataset_version_id: str | None = None,
        max_examples: int | None = None,
        example_ids: builtins.list[str] | None = None,
        evaluation_task_ids: builtins.list[str] | None = None,
        tracing_metadata: dict[str, Any] | None = None,
    ) -> TaskRun:
        """Trigger an on-demand run for a task.

        Dispatches based on the task's type — resolves the task by ID or name
        first, then GETs it to determine whether it is an evaluation task or a
        ``run_experiment`` task, and builds the appropriate trigger body.

        For **evaluation tasks** (``template_evaluation`` /
        ``code_evaluation``):

        - Valid fields: ``data_start_time``, ``data_end_time``, ``max_spans``,
          ``override_evaluations``, ``experiment_ids``.
        - All fields are optional; an empty trigger body uses server defaults.

        For **run_experiment tasks**:

        - Valid fields: ``experiment_name`` (**required**),
          ``dataset_version_id``, ``max_examples``, ``example_ids``,
          ``tracing_metadata``, ``evaluation_task_ids``.
        - ``experiment_name`` is the display name for the new experiment that
          will be created for this run.
        - ``example_ids`` and ``max_examples`` are mutually exclusive; at most
          one may be provided. When both are omitted, all examples are used.

        Args:
            task: Task name or global ID (base64) to trigger a run for.
            space: Optional space name or ID used to disambiguate the task
                lookup. Recommended when resolving by name.
            data_start_time: Start of the data window to evaluate. Evaluation
                tasks only.
            data_end_time: End of the data window to evaluate. Defaults to now
                when omitted. Evaluation tasks only.
            max_spans: Maximum number of spans to process (default 10 000).
                Evaluation tasks only.
            override_evaluations: Whether to re-evaluate data that already has
                evaluation labels. Defaults to ``False``. Evaluation tasks
                only.
            experiment_ids: Experiment global IDs (base64) to run against.
                Only applicable for dataset-based evaluation tasks.
            experiment_name: Display name for the experiment to be created.
                Must be unique within the dataset. Required for
                ``run_experiment`` tasks.
            dataset_version_id: Dataset version global ID (base64). Defaults
                to the latest version when omitted. ``run_experiment`` tasks
                only.
            max_examples: Maximum number of examples to run (dataset order).
                Mutually exclusive with ``example_ids``. When both are
                omitted, all examples are used. ``run_experiment`` tasks only.
            example_ids: Specific dataset example global IDs (base64) to run
                against. Mutually exclusive with ``max_examples``. When both
                are omitted, all examples are used. ``run_experiment`` tasks
                only.
            tracing_metadata: Arbitrary key-value metadata attached to the
                run's traces. ``run_experiment`` tasks only.
            evaluation_task_ids: Task global IDs (base64) of evaluation tasks
                to trigger after the experiment run completes. Supported for
                all ``run_experiment`` experiment types. ``run_experiment``
                tasks only.

        Returns:
            The newly created task run (initially in ``"pending"`` status).

        Raises:
            ValueError: If a field is not valid for the resolved task type, if
                ``experiment_name`` is missing for a ``run_experiment`` task,
                or if both ``example_ids`` and ``max_examples`` are provided.
            ApiException: If the API request fails.
        """
        from arize._generated import api_client as gen

        task_id = _find_task_id(
            api=self._api,
            task=task,
            space=space,
        )
        task_obj = self._api.tasks_get(task_id=task_id)

        if task_obj.type == "run_experiment":
            eval_only_supplied = {
                k: v
                for k, v in {
                    "data_start_time": data_start_time,
                    "data_end_time": data_end_time,
                    "max_spans": max_spans,
                    "override_evaluations": override_evaluations,
                    "experiment_ids": experiment_ids,
                }.items()
                if v is not None
            }
            if eval_only_supplied:
                raise ValueError(
                    "Fields not valid for run_experiment tasks: "
                    f"{', '.join(eval_only_supplied)}. "
                    "Use 'experiment_name', 'dataset_version_id', "
                    "'max_examples', 'example_ids', 'tracing_metadata', "
                    "or 'evaluation_task_ids' instead.",
                )
            if experiment_name is None:
                raise ValueError(
                    "'experiment_name' is required when triggering a "
                    "run_experiment task.",
                )
            if example_ids is not None and max_examples is not None:
                raise ValueError(
                    "'example_ids' and 'max_examples' are mutually exclusive; "
                    "provide at most one.",
                )
            inner_run_exp = gen.TriggerRunExperimentTaskRunRequest(
                experiment_name=experiment_name,
                dataset_version_id=dataset_version_id,
                max_examples=max_examples,
                example_ids=example_ids,
                tracing_metadata=tracing_metadata,
                evaluation_task_ids=evaluation_task_ids,
            )
            body = gen.TasksTriggerRunRequest(actual_instance=inner_run_exp)
        else:
            run_exp_only_supplied = {
                k: v
                for k, v in {
                    "experiment_name": experiment_name,
                    "dataset_version_id": dataset_version_id,
                    "max_examples": max_examples,
                    "example_ids": example_ids,
                    "tracing_metadata": tracing_metadata,
                    "evaluation_task_ids": evaluation_task_ids,
                }.items()
                if v is not None
            }
            if run_exp_only_supplied:
                raise ValueError(
                    f"Fields not valid for '{task_obj.type}' tasks: "
                    f"{', '.join(run_exp_only_supplied)}. "
                    "Use 'data_start_time', 'data_end_time', 'max_spans', "
                    "'override_evaluations', or 'experiment_ids' instead.",
                )
            inner = gen.TriggerEvaluationTaskRunRequest(
                data_start_time=data_start_time,
                data_end_time=data_end_time,
                max_spans=max_spans,
                override_evaluations=override_evaluations,
                experiment_ids=experiment_ids,
            )
            body = gen.TasksTriggerRunRequest(actual_instance=inner)

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
