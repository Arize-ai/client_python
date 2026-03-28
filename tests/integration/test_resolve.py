"""Integration tests for resolve functions against the real Arize API.

These tests are skipped unless both ARIZE_API_KEY and ARIZE_TEST_SPACE_NAME
environment variables are set. They verify that every find/resolve function can
look up real resources by name and that ID passthrough works correctly.

``ARIZE_TEST_SPACE_NAME`` may be a space **name** or a base64 GraphQL space
global ID. List endpoints filter by ``space_name`` (substring on the
human-readable name only); when the env value is an ID, tests use ``space_id``
instead so resources are found.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_resolve.py -m integration -v
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from arize.utils.resolve import (
    ResolutionError,
    ResolvedIdentifier,
    find_ai_integration_id,
    find_annotation_config_id,
    find_dataset_id,
    find_evaluator_id,
    find_experiment_id,
    find_project_id,
    find_prompt_id,
    find_space_id,
    find_task_id,
    is_resource_id,
    resolve_resource,
)

if TYPE_CHECKING:
    from arize._generated.api_client import SpacesApi

# ---------------------------------------------------------------------------
# Skip the entire module when credentials are not available
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("ARIZE_API_KEY", "")
SPACE_NAME = os.environ.get("ARIZE_TEST_SPACE_NAME", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY or not SPACE_NAME,
        reason="ARIZE_API_KEY and ARIZE_TEST_SPACE_NAME must be set",
    ),
]


def _space_filter_kwargs(spaces_api: SpacesApi) -> dict[str, Any]:
    """``space_id`` or ``space_name`` for list/find APIs from ``SPACE_NAME``."""
    if is_resource_id(SPACE_NAME):
        return {"space_id": find_space_id(spaces_api, SPACE_NAME)}
    return {"space_name": SPACE_NAME}


def _resolve_space(spaces_api: SpacesApi) -> ResolvedIdentifier:
    """Return a ResolvedIdentifier for the test space from ``SPACE_NAME``."""
    if is_resource_id(SPACE_NAME):
        return ResolvedIdentifier(id=find_space_id(spaces_api, SPACE_NAME))
    return ResolvedIdentifier(name=SPACE_NAME)


# ---------------------------------------------------------------------------
# Fixture: shared generated API client
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def generated_client() -> Any:
    """Create a real generated API client for integration tests."""
    from arize.config import SDKConfiguration

    config = SDKConfiguration(api_key=API_KEY)
    from arize._generated.api_client.api_client import ApiClient
    from arize._generated.api_client.configuration import Configuration

    # Must set access_token: generated Configuration.auth_settings() only
    # adds Bearer auth from access_token (see _client_factory.py), not api_key dict.
    api_config = Configuration(host=config.api_url, access_token=API_KEY)
    return ApiClient(api_config)


@pytest.fixture(scope="module")
def spaces_api(generated_client) -> SpacesApi:
    """Create a SpacesApi instance for integration tests."""
    from arize._generated import api_client as gen

    return gen.SpacesApi(generated_client)


# ---------------------------------------------------------------------------
# resolve_resource / find_space_id
# ---------------------------------------------------------------------------
class TestResolveResource:
    """Tests for resolve_resource() and find_space_id()."""

    def test_resolve_resource_by_name(self) -> None:
        """resolve_resource maps IDs to ResourceIDOrName(id=...) and names to ResourceIDOrName(name=...)."""
        r = resolve_resource(SPACE_NAME)
        if is_resource_id(SPACE_NAME):
            assert r.id == SPACE_NAME
            assert r.name is None
        else:
            assert r.id is None
            assert r.name == SPACE_NAME

    def test_find_space_id_by_name(self, spaces_api) -> None:
        """find_space_id resolves a name to an ID or returns an ID as-is."""
        space_id = find_space_id(spaces_api, SPACE_NAME)
        assert is_resource_id(space_id)

    def test_find_space_id_passthrough(self, spaces_api) -> None:
        """find_space_id returns a base64 ID as-is."""
        real_id = find_space_id(spaces_api, SPACE_NAME)
        assert find_space_id(spaces_api, real_id) == real_id

    def test_find_space_id_not_found(self, spaces_api) -> None:
        """find_space_id raises ResolutionError for unknown names."""
        with pytest.raises(ResolutionError, match="space"):
            find_space_id(spaces_api, "nonexistent-space-abc-xyz-12345")


# ---------------------------------------------------------------------------
# find_project_id
# ---------------------------------------------------------------------------
class TestFindProjectId:
    """Tests for find_project_id with space_name passthrough."""

    @pytest.fixture(scope="class")
    def projects_api(self, generated_client) -> Any:
        from arize._generated import api_client as gen

        return gen.ProjectsApi(generated_client)

    @pytest.fixture(scope="class")
    def project_info(self, generated_client, spaces_api) -> dict[str, Any]:
        """Fetch a real project from the test space."""
        from arize._generated import api_client as gen

        api = gen.ProjectsApi(generated_client)
        resp = api.projects_list(**_space_filter_kwargs(spaces_api), limit=1)
        if not resp.projects:
            pytest.skip("No projects in test space")
        p = resp.projects[0]
        return {"id": p.id, "name": p.name}

    def test_resolve_by_name_with_space_name(
        self, projects_api, spaces_api, project_info: dict[str, Any]
    ) -> None:
        """Resolves project name using space_name or space_id from the env."""
        result = find_project_id(
            projects_api,
            project_info["name"],
            _resolve_space(spaces_api),
        )
        assert result == project_info["id"]

    def test_resolve_by_name_with_space_id(
        self, projects_api, spaces_api, project_info: dict[str, Any]
    ) -> None:
        """Resolves project name using space_id."""
        space_id = find_space_id(spaces_api, SPACE_NAME)
        result = find_project_id(
            projects_api,
            project_info["name"],
            ResolvedIdentifier(id=space_id),
        )
        assert result == project_info["id"]

    def test_id_passthrough(
        self, projects_api, project_info: dict[str, Any]
    ) -> None:
        """A base64 ID is returned as-is without any API call."""
        result = find_project_id(projects_api, project_info["id"])
        assert result == project_info["id"]

    def test_not_found(self, projects_api, spaces_api) -> None:
        """Raises ResolutionError for unknown project name."""
        with pytest.raises(ResolutionError, match="project"):
            find_project_id(
                projects_api,
                "nonexistent-project-abc-xyz-12345",
                _resolve_space(spaces_api),
            )


# ---------------------------------------------------------------------------
# find_dataset_id
# ---------------------------------------------------------------------------
class TestFindDatasetId:
    """Tests for find_dataset_id with space_name passthrough."""

    @pytest.fixture(scope="class")
    def datasets_api(self, generated_client) -> Any:
        from arize._generated import api_client as gen

        return gen.DatasetsApi(generated_client)

    @pytest.fixture(scope="class")
    def dataset_info(self, generated_client, spaces_api) -> dict[str, Any]:
        """Fetch a real dataset from the test space."""
        from arize._generated import api_client as gen

        api = gen.DatasetsApi(generated_client)
        resp = api.datasets_list(**_space_filter_kwargs(spaces_api), limit=1)
        if not resp.datasets:
            pytest.skip("No datasets in test space")
        d = resp.datasets[0]
        return {"id": d.id, "name": d.name}

    def test_resolve_by_name_with_space_name(
        self, datasets_api, spaces_api, dataset_info: dict[str, Any]
    ) -> None:
        """Resolves dataset name using space_name or space_id from the env."""
        result = find_dataset_id(
            datasets_api,
            dataset_info["name"],
            _resolve_space(spaces_api),
        )
        assert result == dataset_info["id"]

    def test_id_passthrough(
        self, datasets_api, dataset_info: dict[str, Any]
    ) -> None:
        """A base64 ID is returned as-is."""
        result = find_dataset_id(datasets_api, dataset_info["id"])
        assert result == dataset_info["id"]

    def test_not_found(self, datasets_api, spaces_api) -> None:
        """Raises ResolutionError for unknown dataset name."""
        with pytest.raises(ResolutionError, match="dataset"):
            find_dataset_id(
                datasets_api,
                "nonexistent-dataset-abc-xyz-12345",
                _resolve_space(spaces_api),
            )


# ---------------------------------------------------------------------------
# find_prompt_id
# ---------------------------------------------------------------------------
class TestFindPromptId:
    """Tests for find_prompt_id with space_name passthrough."""

    @pytest.fixture(scope="class")
    def prompts_api(self, generated_client) -> Any:
        from arize._generated import api_client as gen

        return gen.PromptsApi(generated_client)

    @pytest.fixture(scope="class")
    def prompt_info(self, generated_client, spaces_api) -> dict[str, Any]:
        """Fetch a real prompt from the test space."""
        from arize._generated import api_client as gen

        api = gen.PromptsApi(generated_client)
        resp = api.prompts_list(**_space_filter_kwargs(spaces_api), limit=1)
        if not resp.prompts:
            pytest.skip("No prompts in test space")
        p = resp.prompts[0]
        return {"id": p.id, "name": p.name}

    def test_resolve_by_name_with_space_name(
        self, prompts_api, spaces_api, prompt_info: dict[str, Any]
    ) -> None:
        """Resolves prompt name using space_name or space_id from the env."""
        result = find_prompt_id(
            prompts_api,
            prompt_info["name"],
            _resolve_space(spaces_api),
        )
        assert result == prompt_info["id"]

    def test_id_passthrough(
        self, prompts_api, prompt_info: dict[str, Any]
    ) -> None:
        """A base64 ID is returned as-is."""
        result = find_prompt_id(prompts_api, prompt_info["id"])
        assert result == prompt_info["id"]


# ---------------------------------------------------------------------------
# find_evaluator_id
# ---------------------------------------------------------------------------
class TestFindEvaluatorId:
    """Tests for find_evaluator_id with space_name passthrough."""

    @pytest.fixture(scope="class")
    def evaluators_api(self, generated_client) -> Any:
        from arize._generated import api_client as gen

        return gen.EvaluatorsApi(generated_client)

    @pytest.fixture(scope="class")
    def evaluator_info(self, generated_client, spaces_api) -> dict[str, Any]:
        """Fetch a real evaluator from the test space."""
        from arize._generated import api_client as gen

        api = gen.EvaluatorsApi(generated_client)
        resp = api.evaluators_list(**_space_filter_kwargs(spaces_api), limit=1)
        if not resp.evaluators:
            pytest.skip("No evaluators in test space")
        e = resp.evaluators[0]
        return {"id": e.id, "name": e.name}

    def test_resolve_by_name_with_space_name(
        self, evaluators_api, spaces_api, evaluator_info: dict[str, Any]
    ) -> None:
        """Resolves evaluator name using space_name or space_id from the env."""
        result = find_evaluator_id(
            evaluators_api,
            evaluator_info["name"],
            _resolve_space(spaces_api),
        )
        assert result == evaluator_info["id"]

    def test_id_passthrough(
        self, evaluators_api, evaluator_info: dict[str, Any]
    ) -> None:
        """A base64 ID is returned as-is."""
        result = find_evaluator_id(evaluators_api, evaluator_info["id"])
        assert result == evaluator_info["id"]


# ---------------------------------------------------------------------------
# find_annotation_config_id
# ---------------------------------------------------------------------------
class TestFindAnnotationConfigId:
    """Tests for find_annotation_config_id with space_name passthrough."""

    @pytest.fixture(scope="class")
    def annotation_configs_api(self, generated_client) -> Any:
        from arize._generated import api_client as gen

        return gen.AnnotationConfigsApi(generated_client)

    @pytest.fixture(scope="class")
    def annotation_config_info(
        self, generated_client, spaces_api
    ) -> dict[str, Any]:
        """Fetch a real annotation config from the test space."""
        from arize._generated import api_client as gen

        api = gen.AnnotationConfigsApi(generated_client)
        resp = api.annotation_configs_list(
            **_space_filter_kwargs(spaces_api), limit=1
        )
        if not resp.annotation_configs:
            pytest.skip("No annotation configs in test space")
        ac = resp.annotation_configs[0]
        inner = ac.actual_instance
        assert inner is not None
        return {"id": inner.id, "name": inner.name}

    def test_resolve_by_name_with_space_name(
        self,
        annotation_configs_api,
        spaces_api,
        annotation_config_info: dict[str, Any],
    ) -> None:
        """Resolves annotation config name using space_name or space_id."""
        result = find_annotation_config_id(
            annotation_configs_api,
            annotation_config_info["name"],
            _resolve_space(spaces_api),
        )
        assert result == annotation_config_info["id"]

    def test_id_passthrough(
        self,
        annotation_configs_api,
        annotation_config_info: dict[str, Any],
    ) -> None:
        """A base64 ID is returned as-is."""
        result = find_annotation_config_id(
            annotation_configs_api, annotation_config_info["id"]
        )
        assert result == annotation_config_info["id"]


# ---------------------------------------------------------------------------
# find_experiment_id
# ---------------------------------------------------------------------------
class TestFindExperimentId:
    """Tests for find_experiment_id (requires dataset_id, not space)."""

    @pytest.fixture(scope="class")
    def experiments_api(self, generated_client) -> Any:
        from arize._generated import api_client as gen

        return gen.ExperimentsApi(generated_client)

    @pytest.fixture(scope="class")
    def experiment_info(self, generated_client, spaces_api) -> dict[str, Any]:
        """Fetch a real experiment via a dataset in the test space."""
        from arize._generated import api_client as gen

        datasets_api = gen.DatasetsApi(generated_client)
        ds_resp = datasets_api.datasets_list(
            **_space_filter_kwargs(spaces_api), limit=10
        )
        if not ds_resp.datasets:
            pytest.skip("No datasets in test space")

        experiments_api = gen.ExperimentsApi(generated_client)
        for ds in ds_resp.datasets:
            exp_resp = experiments_api.experiments_list(
                dataset_id=ds.id, limit=1
            )
            if exp_resp.experiments:
                e = exp_resp.experiments[0]
                return {
                    "id": e.id,
                    "name": e.name,
                    "dataset_id": ds.id,
                }
        pytest.skip("No experiments found in test space datasets")

    def test_resolve_by_name(
        self, experiments_api, experiment_info: dict[str, Any]
    ) -> None:
        """Resolves experiment name using dataset_id."""
        result = find_experiment_id(
            experiments_api,
            experiment_info["name"],
            dataset_id=experiment_info["dataset_id"],
        )
        assert result == experiment_info["id"]

    def test_id_passthrough(
        self, experiments_api, experiment_info: dict[str, Any]
    ) -> None:
        """A base64 ID is returned as-is."""
        result = find_experiment_id(experiments_api, experiment_info["id"])
        assert result == experiment_info["id"]


# ---------------------------------------------------------------------------
# find_task_id
# ---------------------------------------------------------------------------
class TestFindTaskId:
    """Tests for find_task_id with space_name passthrough."""

    @pytest.fixture(scope="class")
    def tasks_api(self, generated_client) -> Any:
        from arize._generated import api_client as gen

        return gen.TasksApi(generated_client)

    @pytest.fixture(scope="class")
    def task_info(self, generated_client, spaces_api) -> dict[str, Any]:
        """Fetch a real task from the test space."""
        from arize._generated import api_client as gen

        api = gen.TasksApi(generated_client)
        resp = api.tasks_list(**_space_filter_kwargs(spaces_api), limit=1)
        if not resp.tasks:
            pytest.skip("No tasks in test space")
        t = resp.tasks[0]
        return {"id": t.id, "name": t.name}

    def test_resolve_by_name_with_space_name(
        self, tasks_api, spaces_api, task_info: dict[str, Any]
    ) -> None:
        """Resolves task name using space_name or space_id from the env."""
        result = find_task_id(
            tasks_api,
            task_info["name"],
            _resolve_space(spaces_api),
        )
        assert result == task_info["id"]

    def test_id_passthrough(self, tasks_api, task_info: dict[str, Any]) -> None:
        """A base64 ID is returned as-is."""
        result = find_task_id(tasks_api, task_info["id"])
        assert result == task_info["id"]


# ---------------------------------------------------------------------------
# find_ai_integration_id
# ---------------------------------------------------------------------------
class TestFindAiIntegrationId:
    """Tests for find_ai_integration_id with space_name passthrough."""

    @pytest.fixture(scope="class")
    def ai_integrations_api(self, generated_client) -> Any:
        from arize._generated import api_client as gen

        return gen.AIIntegrationsApi(generated_client)

    @pytest.fixture(scope="class")
    def ai_integration_info(
        self, generated_client, spaces_api
    ) -> dict[str, Any]:
        """Fetch a real AI integration from the test space."""
        from arize._generated import api_client as gen

        api = gen.AIIntegrationsApi(generated_client)
        resp = api.ai_integrations_list(
            **_space_filter_kwargs(spaces_api), limit=1
        )
        if not resp.ai_integrations:
            pytest.skip("No AI integrations in test space")
        ai = resp.ai_integrations[0]
        return {"id": ai.id, "name": ai.name}

    def test_resolve_by_name_with_space_name(
        self,
        ai_integrations_api,
        spaces_api,
        ai_integration_info: dict[str, Any],
    ) -> None:
        """Resolves AI integration name using space_name or space_id."""
        result = find_ai_integration_id(
            ai_integrations_api,
            ai_integration_info["name"],
            _resolve_space(spaces_api),
        )
        assert result == ai_integration_info["id"]

    def test_id_passthrough(
        self,
        ai_integrations_api,
        ai_integration_info: dict[str, Any],
    ) -> None:
        """A base64 ID is returned as-is."""
        result = find_ai_integration_id(
            ai_integrations_api, ai_integration_info["id"]
        )
        assert result == ai_integration_info["id"]
