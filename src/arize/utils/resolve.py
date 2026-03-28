"""Utilities for resolving resource identifiers (ID or name) to IDs."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arize._generated.api_client import (
        AIIntegrationsApi,
        AnnotationConfigsApi,
        DatasetsApi,
        EvaluatorsApi,
        ExperimentsApi,
        ProjectsApi,
        PromptsApi,
        SpacesApi,
        TasksApi,
    )

logger = logging.getLogger(__name__)


class ResolutionError(Exception):
    """Raised when a resource name cannot be resolved to an ID."""

    def __init__(  # noqa:D107
        self,
        resource_type: str,
        name: str,
        available_names: list[str] | None = None,
        hint: str | None = None,
    ) -> None:
        available = ""
        if available_names:
            available = (
                f" Available {resource_type}s: {', '.join(available_names)}"
            )
        hint_text = f" {hint}" if hint else ""
        super().__init__(
            f"{resource_type} '{name}' not found.{available}{hint_text}"
        )
        self.resource_type = resource_type
        self.resource_name = name
        self.available_names = available_names or []


@dataclass
class ResolvedIdentifier:
    """Holds either a resource ID or a resource name (not both)."""

    id: str | None = None
    name: str | None = None

    def is_id(self) -> bool:
        """Return True if this holds an ID."""
        return self.id is not None

    def is_name(self) -> bool:
        """Return True if this holds a name."""
        return self.name is not None

    def is_set(self) -> bool:
        """Return True if this holds either an ID or a name."""
        return self.is_id() or self.is_name()


def is_resource_id(value: str) -> bool:
    """Return True if *value* looks like a base64-encoded resource ID.

    Base64 IDs decode to a string containing a colon, e.g. ``"Space:1:bFuL"``.
    """
    try:
        decoded = base64.b64decode(value, validate=True).decode("utf-8")
    except Exception:
        return False
    return ":" in decoded


def resolve_resource(value: str | None) -> ResolvedIdentifier:
    """Split a name-or-ID string into a :class:`ResolvedIdentifier`.

    - ``None`` → both fields ``None``
    - base64-encoded global ID → ``id`` field set
    - any other string → ``name`` field set

    Args:
        value: A resource ID, resource name, or *None*.

    Returns:
        A :class:`ResolvedIdentifier` instance.
    """
    if value is None:
        return ResolvedIdentifier()
    if is_resource_id(value):
        return ResolvedIdentifier(id=value)
    return ResolvedIdentifier(name=value)


def find_space_id(api: SpacesApi, space: str) -> str:
    """Resolve a space ID or name to a space ID.

    If *space* is a base64-encoded global ID it is returned as-is. Otherwise,
    the list spaces endpoint is called to find an exact name match.

    Args:
        api: SpacesApi instance.
        space: Space ID or name.

    Returns:
        The resolved space ID.

    Raises:
        ResolutionError: If the space name cannot be found.
    """
    if is_resource_id(space):
        return space

    available: list[str] = []
    cursor: str | None = None

    while True:
        response = api.spaces_list(
            name=space,
            limit=100,
            cursor=cursor,
        )
        for s in response.spaces:
            if s.name == space:
                logger.debug("Resolved space '%s' → %s", space, s.id)
                return s.id
            available.append(s.name)
        cursor = getattr(response.pagination, "next_cursor", None)
        if not cursor:
            break

    raise ResolutionError("space", space, available)


def find_project_id(
    api: ProjectsApi,
    project: str,
    space: str | None,
) -> str:
    """Resolve a project ID or name to a project ID.

    Args:
        api: ProjectsApi instance.
        project: Project ID or name.
        space: Space ID or name used to filter the lookup.

    Returns:
        The resolved project ID.

    Raises:
        ResolutionError: If the project name cannot be found or *space* is not
            provided when needed.
    """
    if is_resource_id(project):
        return project

    resolved_space = resolve_resource(space)
    if not resolved_space.is_set():
        raise ResolutionError(
            "project",
            project,
            hint=(
                "Provide 'space' so the project name can be resolved, "
                "or provide the project ID instead of the name."
            ),
        )

    available: list[str] = []
    cursor: str | None = None

    while True:
        response = api.projects_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=project,
            limit=100,
            cursor=cursor,
        )
        for p in response.projects:
            if p.name == project:
                logger.debug("Resolved project '%s' → %s", project, p.id)
                return p.id
            available.append(p.name)
        cursor = getattr(response.pagination, "next_cursor", None)
        if not cursor:
            break

    raise ResolutionError("project", project, available)


def find_dataset_id(
    api: DatasetsApi,
    dataset: str,
    space: str | None,
) -> str:
    """Resolve a dataset ID or name to a dataset ID.

    Args:
        api: DatasetsApi instance.
        dataset: Dataset ID or name.
        space: Space ID or name used to filter the lookup.

    Returns:
        The resolved dataset ID.

    Raises:
        ResolutionError: If the dataset name cannot be found.
    """
    if is_resource_id(dataset):
        return dataset

    resolved_space = resolve_resource(space)
    if not resolved_space.is_set():
        raise ResolutionError(
            "dataset",
            dataset,
            hint=(
                "Provide 'space' so the dataset name can be resolved, "
                "or provide the dataset ID instead of the name."
            ),
        )

    available: list[str] = []
    cursor: str | None = None

    while True:
        response = api.datasets_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=dataset,
            limit=100,
            cursor=cursor,
        )
        for d in response.datasets:
            if d.name == dataset:
                logger.debug("Resolved dataset '%s' → %s", dataset, d.id)
                return d.id
            available.append(d.name)
        cursor = getattr(response.pagination, "next_cursor", None)
        if not cursor:
            break

    raise ResolutionError("dataset", dataset, available)


def find_experiment_id(
    api: ExperimentsApi,
    datasets_api: DatasetsApi,
    experiment: str,
    dataset: str | None,
    space: str | None,
) -> str:
    """Resolve an experiment ID or name to an experiment ID.

    Args:
        api: ExperimentsApi instance.
        datasets_api: DatasetsApi instance, used to resolve a dataset name to an ID.
        experiment: Experiment ID or name.
        dataset: Dataset ID or name. Required when *experiment* is a name.
        space: Space ID or name used to resolve *dataset* by name.

    Returns:
        The resolved experiment ID.

    Raises:
        ResolutionError: If the experiment name cannot be found.
    """
    if is_resource_id(experiment):
        return experiment

    resolved_dataset = resolve_resource(dataset)
    resolved_space = resolve_resource(space)

    if not resolved_dataset.is_set():
        raise ResolutionError(
            "experiment",
            experiment,
            hint=(
                "Provide 'dataset' so the experiment name can be resolved, "
                "or provide the experiment ID instead of the name."
            ),
        )

    if resolved_dataset.is_name() and not resolved_space.is_set():
        raise ResolutionError(
            "experiment",
            experiment,
            hint=(
                "Provide 'space' so the dataset name can be resolved, "
                "which is needed to resolve the experiment name. Alternatively, "
                "you can provide the experiment ID, or the dataset ID instead of the name."
            ),
        )

    dataset_id = (
        resolved_dataset.id
        if resolved_dataset.is_id()
        else find_dataset_id(datasets_api, resolved_dataset.name, space)  # type:ignore
    )

    available: list[str] = []
    cursor: str | None = None

    while True:
        response = api.experiments_list(
            dataset_id=dataset_id,
            limit=100,
            cursor=cursor,
        )
        for e in response.experiments:
            if e.name == experiment:
                logger.debug("Resolved experiment '%s' → %s", experiment, e.id)
                return e.id
            available.append(e.name)
        cursor = getattr(response.pagination, "next_cursor", None)
        if not cursor:
            break

    raise ResolutionError("experiment", experiment, available)


def find_prompt_id(
    api: PromptsApi,
    prompt: str,
    space: str | None,
) -> str:
    """Resolve a prompt ID or name to a prompt ID.

    Args:
        api: PromptsApi instance.
        prompt: Prompt ID or name.
        space: Space ID or name used to filter the lookup.

    Returns:
        The resolved prompt ID.

    Raises:
        ResolutionError: If the prompt name cannot be found.
    """
    if is_resource_id(prompt):
        return prompt

    resolved_space = resolve_resource(space)
    if not resolved_space.is_set():
        raise ResolutionError(
            "prompt",
            prompt,
            hint=(
                "Provide 'space' so the prompt name can be resolved, "
                "or provide the prompt ID instead of the name."
            ),
        )

    available: list[str] = []
    cursor: str | None = None

    while True:
        response = api.prompts_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=prompt,
            limit=100,
            cursor=cursor,
        )
        for p in response.prompts:
            if p.name == prompt:
                logger.debug("Resolved prompt '%s' → %s", prompt, p.id)
                return p.id
            available.append(p.name)
        cursor = getattr(response.pagination, "next_cursor", None)
        if not cursor:
            break

    raise ResolutionError("prompt", prompt, available)


def find_evaluator_id(
    api: EvaluatorsApi,
    evaluator: str,
    space: str | None,
) -> str:
    """Resolve an evaluator ID or name to an evaluator ID.

    Args:
        api: EvaluatorsApi instance.
        evaluator: Evaluator ID or name.
        space: Space ID or name used to filter the lookup.

    Returns:
        The resolved evaluator ID.

    Raises:
        ResolutionError: If the evaluator name cannot be found.
    """
    if is_resource_id(evaluator):
        return evaluator

    resolved_space = resolve_resource(space)
    if not resolved_space.is_set():
        raise ResolutionError(
            "evaluator",
            evaluator,
            hint=(
                "Provide 'space' so the evaluator name can be resolved, "
                "or provide the evaluator ID instead of the name."
            ),
        )

    available: list[str] = []
    cursor: str | None = None

    while True:
        response = api.evaluators_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=evaluator,
            limit=100,
            cursor=cursor,
        )
        for ev in response.evaluators:
            if ev.name == evaluator:
                logger.debug("Resolved evaluator '%s' → %s", evaluator, ev.id)
                return ev.id
            available.append(ev.name)
        cursor = getattr(response.pagination, "next_cursor", None)
        if not cursor:
            break

    raise ResolutionError("evaluator", evaluator, available)


def find_annotation_config_id(
    api: AnnotationConfigsApi,
    annotation_config: str,
    space: str | None,
) -> str:
    """Resolve an annotation config ID or name to an annotation config ID.

    Args:
        api: AnnotationConfigsApi instance.
        annotation_config: Annotation config ID or name.
        space: Space ID or name used to filter the lookup.

    Returns:
        The resolved annotation config ID.

    Raises:
        ResolutionError: If the annotation config name cannot be found.
    """
    if is_resource_id(annotation_config):
        return annotation_config

    resolved_space = resolve_resource(space)
    if not resolved_space.is_set():
        raise ResolutionError(
            "annotation config",
            annotation_config,
            hint=(
                "Provide 'space' so the annotation config name can be resolved, "
                "or provide the annotation config ID instead of the name."
            ),
        )

    available: list[str] = []
    cursor: str | None = None

    while True:
        response = api.annotation_configs_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=annotation_config,
            limit=100,
            cursor=cursor,
        )
        for ac in response.annotation_configs:
            inner = ac.actual_instance
            if inner is None:
                continue
            if inner.name == annotation_config:
                logger.debug(
                    "Resolved annotation config '%s' → %s",
                    annotation_config,
                    inner.id,
                )
                return inner.id
            available.append(inner.name)
        cursor = getattr(response.pagination, "next_cursor", None)
        if not cursor:
            break

    raise ResolutionError("annotation config", annotation_config, available)


def find_ai_integration_id(
    api: AIIntegrationsApi,
    integration: str,
    space: str | None,
) -> str:
    """Resolve an AI integration ID or name to an AI integration ID.

    Args:
        api: AIIntegrationsApi instance.
        integration: AI integration ID or name.
        space: Space ID or name used to filter the lookup.

    Returns:
        The resolved AI integration ID.

    Raises:
        ResolutionError: If the AI integration name cannot be found.
    """
    if is_resource_id(integration):
        return integration

    resolved_space = resolve_resource(space)
    if not resolved_space.is_set():
        raise ResolutionError(
            "AI integration",
            integration,
            hint=(
                "Provide 'space' so the AI integration name can be resolved, "
                "or provide the AI integration ID instead of the name."
            ),
        )

    available: list[str] = []
    cursor: str | None = None

    while True:
        response = api.ai_integrations_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=integration,
            limit=100,
            cursor=cursor,
        )
        for ai in response.ai_integrations:
            if ai.name == integration:
                logger.debug(
                    "Resolved AI integration '%s' → %s", integration, ai.id
                )
                return ai.id
            available.append(ai.name)
        cursor = getattr(response.pagination, "next_cursor", None)
        if not cursor:
            break

    raise ResolutionError("AI integration", integration, available)


def find_task_id(
    api: TasksApi,
    task: str,
    space: str | None,
) -> str:
    """Resolve a task ID or name to a task ID.

    Args:
        api: TasksApi instance.
        task: Task ID or name.
        space: Space ID or name used to filter the lookup.

    Returns:
        The resolved task ID.

    Raises:
        ResolutionError: If the task name cannot be found.
    """
    if is_resource_id(task):
        return task

    resolved_space = resolve_resource(space)
    if not resolved_space.is_set():
        raise ResolutionError(
            "task",
            task,
            hint=(
                "Provide 'space' so the task name can be resolved, "
                "or provide the task ID instead of the name."
            ),
        )

    available: list[str] = []
    cursor: str | None = None

    while True:
        response = api.tasks_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=task,
            limit=100,
            cursor=cursor,
        )
        for t in response.tasks:
            if t.name == task:
                logger.debug("Resolved task '%s' → %s", task, t.id)
                return t.id
            available.append(t.name)
        cursor = getattr(response.pagination, "next_cursor", None)
        if not cursor:
            break

    raise ResolutionError("task", task, available)
