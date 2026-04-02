"""Integration tests for resolve functions against the real Arize API.

These tests are skipped unless both ARIZE_API_KEY and ARIZE_TEST_SPACE_NAME
environment variables are set. They verify that every find/resolve function can
look up real resources by name and that ID passthrough works correctly.

``ARIZE_TEST_SPACE_NAME`` may be a space **name** or a base64 GraphQL space
global ID. The find functions accept either format — a base64-encoded value is
treated as a space ID, any other string is treated as a name.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_resolve.py -m integration -v
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from arize.utils.resolve import (
    NotFoundError,
    _find_ai_integration_id,
    _find_annotation_config_id,
    _find_dataset_id,
    _find_evaluator_id,
    _find_experiment_id,
    _find_project_id,
    _find_prompt_id,
    _find_role_id,
    _find_space_id,
    _find_task_id,
    _resolve_resource,
    is_resource_id,
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
        return {"space_id": _find_space_id(spaces_api, SPACE_NAME)}
    return {"space_name": SPACE_NAME}


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
# _resolve_resource / _find_space_id
# ---------------------------------------------------------------------------
class TestResolveResource:
    """Tests for _resolve_resource() and _find_space_id()."""

    def test_resolve_resource_by_name(self) -> None:
        """_resolve_resource maps IDs to ResourceIDOrName(id=...) and names to ResourceIDOrName(name=...)."""
        r = _resolve_resource(SPACE_NAME)
        if is_resource_id(SPACE_NAME):
            assert r.id == SPACE_NAME
            assert r.name is None
        else:
            assert r.id is None
            assert r.name == SPACE_NAME

    def test_find_space_id_by_name(self, spaces_api) -> None:
        """_find_space_id resolves a name to an ID or returns an ID as-is."""
        space_id = _find_space_id(spaces_api, SPACE_NAME)
        assert is_resource_id(space_id)

    def test_find_space_id_passthrough(self, spaces_api) -> None:
        """_find_space_id returns a base64 ID as-is."""
        real_id = _find_space_id(spaces_api, SPACE_NAME)
        assert _find_space_id(spaces_api, real_id) == real_id

    def test_find_space_id_not_found(self, spaces_api) -> None:
        """_find_space_id raises NotFoundError for unknown names."""
        with pytest.raises(NotFoundError, match="space"):
            _find_space_id(spaces_api, "nonexistent-space-abc-xyz-12345")


# ---------------------------------------------------------------------------
# _find_project_id
# ---------------------------------------------------------------------------
class TestFindProjectId:
    """Tests for _find_project_id with space_name passthrough."""

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

    def test_resolve_by_name(
        self, projects_api, spaces_api, project_info: dict[str, Any]
    ) -> None:
        """Resolves project name using SPACE_NAME (name or ID string)."""
        result = _find_project_id(
            projects_api,
            project_info["name"],
            SPACE_NAME,
        )
        assert result == project_info["id"]

    def test_resolve_by_name_with_space_id(
        self, projects_api, spaces_api, project_info: dict[str, Any]
    ) -> None:
        """Resolves project name when space is given as an ID string."""
        space_id = _find_space_id(spaces_api, SPACE_NAME)
        result = _find_project_id(
            projects_api,
            project_info["name"],
            space_id,
        )
        assert result == project_info["id"]

    def test_id_passthrough(
        self, projects_api, project_info: dict[str, Any]
    ) -> None:
        """A base64 ID is returned as-is without any API call."""
        result = _find_project_id(projects_api, project_info["id"], None)
        assert result == project_info["id"]

    def test_not_found(self, projects_api, spaces_api) -> None:
        """Raises NotFoundError for unknown project name."""
        with pytest.raises(NotFoundError, match="project"):
            _find_project_id(
                projects_api,
                "nonexistent-project-abc-xyz-12345",
                SPACE_NAME,
            )


# ---------------------------------------------------------------------------
# _find_dataset_id
# ---------------------------------------------------------------------------
class TestFindDatasetId:
    """Tests for _find_dataset_id with space_name passthrough."""

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

    def test_resolve_by_name(
        self, datasets_api, spaces_api, dataset_info: dict[str, Any]
    ) -> None:
        """Resolves dataset name using SPACE_NAME (name or ID string)."""
        result = _find_dataset_id(
            datasets_api,
            dataset_info["name"],
            SPACE_NAME,
        )
        assert result == dataset_info["id"]

    def test_id_passthrough(
        self, datasets_api, dataset_info: dict[str, Any]
    ) -> None:
        """A base64 ID is returned as-is."""
        result = _find_dataset_id(datasets_api, dataset_info["id"], None)
        assert result == dataset_info["id"]

    def test_not_found(self, datasets_api, spaces_api) -> None:
        """Raises NotFoundError for unknown dataset name."""
        with pytest.raises(NotFoundError, match="dataset"):
            _find_dataset_id(
                datasets_api,
                "nonexistent-dataset-abc-xyz-12345",
                SPACE_NAME,
            )


# ---------------------------------------------------------------------------
# _find_prompt_id
# ---------------------------------------------------------------------------
class TestFindPromptId:
    """Tests for _find_prompt_id with space_name passthrough."""

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

    def test_resolve_by_name(
        self, prompts_api, spaces_api, prompt_info: dict[str, Any]
    ) -> None:
        """Resolves prompt name using SPACE_NAME (name or ID string)."""
        result = _find_prompt_id(
            prompts_api,
            prompt_info["name"],
            SPACE_NAME,
        )
        assert result == prompt_info["id"]

    def test_id_passthrough(
        self, prompts_api, prompt_info: dict[str, Any]
    ) -> None:
        """A base64 ID is returned as-is."""
        result = _find_prompt_id(prompts_api, prompt_info["id"], None)
        assert result == prompt_info["id"]


# ---------------------------------------------------------------------------
# _find_evaluator_id
# ---------------------------------------------------------------------------
class TestFindEvaluatorId:
    """Tests for _find_evaluator_id with space_name passthrough."""

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

    def test_resolve_by_name(
        self, evaluators_api, spaces_api, evaluator_info: dict[str, Any]
    ) -> None:
        """Resolves evaluator name using SPACE_NAME (name or ID string)."""
        result = _find_evaluator_id(
            evaluators_api,
            evaluator_info["name"],
            SPACE_NAME,
        )
        assert result == evaluator_info["id"]

    def test_id_passthrough(
        self, evaluators_api, evaluator_info: dict[str, Any]
    ) -> None:
        """A base64 ID is returned as-is."""
        result = _find_evaluator_id(evaluators_api, evaluator_info["id"], None)
        assert result == evaluator_info["id"]


# ---------------------------------------------------------------------------
# _find_annotation_config_id
# ---------------------------------------------------------------------------
class TestFindAnnotationConfigId:
    """Tests for _find_annotation_config_id with space_name passthrough."""

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

    def test_resolve_by_name(
        self,
        annotation_configs_api,
        spaces_api,
        annotation_config_info: dict[str, Any],
    ) -> None:
        """Resolves annotation config name using SPACE_NAME (name or ID string)."""
        result = _find_annotation_config_id(
            annotation_configs_api,
            annotation_config_info["name"],
            SPACE_NAME,
        )
        assert result == annotation_config_info["id"]

    def test_id_passthrough(
        self,
        annotation_configs_api,
        annotation_config_info: dict[str, Any],
    ) -> None:
        """A base64 ID is returned as-is."""
        result = _find_annotation_config_id(
            annotation_configs_api, annotation_config_info["id"], None
        )
        assert result == annotation_config_info["id"]


# ---------------------------------------------------------------------------
# _find_experiment_id
# ---------------------------------------------------------------------------
class TestFindExperimentId:
    """Tests for _find_experiment_id (requires dataset and datasets_api)."""

    @pytest.fixture(scope="class")
    def experiments_api(self, generated_client) -> Any:
        from arize._generated import api_client as gen

        return gen.ExperimentsApi(generated_client)

    @pytest.fixture(scope="class")
    def datasets_api(self, generated_client) -> Any:
        from arize._generated import api_client as gen

        return gen.DatasetsApi(generated_client)

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
        self,
        experiments_api,
        datasets_api,
        experiment_info: dict[str, Any],
    ) -> None:
        """Resolves experiment name using dataset_id (already an ID, no space needed)."""
        result = _find_experiment_id(
            experiments_api,
            datasets_api,
            experiment_info["name"],
            experiment_info["dataset_id"],
            None,
        )
        assert result == experiment_info["id"]

    def test_id_passthrough(
        self,
        experiments_api,
        datasets_api,
        experiment_info: dict[str, Any],
    ) -> None:
        """A base64 ID is returned as-is."""
        result = _find_experiment_id(
            experiments_api,
            datasets_api,
            experiment_info["id"],
            None,
            None,
        )
        assert result == experiment_info["id"]


# ---------------------------------------------------------------------------
# _find_task_id
# ---------------------------------------------------------------------------
class TestFindTaskId:
    """Tests for _find_task_id with space_name passthrough."""

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

    def test_resolve_by_name(
        self, tasks_api, spaces_api, task_info: dict[str, Any]
    ) -> None:
        """Resolves task name using SPACE_NAME (name or ID string)."""
        result = _find_task_id(
            tasks_api,
            task_info["name"],
            SPACE_NAME,
        )
        assert result == task_info["id"]

    def test_id_passthrough(self, tasks_api, task_info: dict[str, Any]) -> None:
        """A base64 ID is returned as-is."""
        result = _find_task_id(tasks_api, task_info["id"], None)
        assert result == task_info["id"]


# ---------------------------------------------------------------------------
# _find_ai_integration_id
# ---------------------------------------------------------------------------
class TestFindAiIntegrationId:
    """Tests for _find_ai_integration_id with space_name passthrough."""

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

    def test_resolve_by_name(
        self,
        ai_integrations_api,
        spaces_api,
        ai_integration_info: dict[str, Any],
    ) -> None:
        """Resolves AI integration name using SPACE_NAME (name or ID string)."""
        result = _find_ai_integration_id(
            ai_integrations_api,
            ai_integration_info["name"],
            SPACE_NAME,
        )
        assert result == ai_integration_info["id"]

    def test_id_passthrough(
        self,
        ai_integrations_api,
        ai_integration_info: dict[str, Any],
    ) -> None:
        """A base64 ID is returned as-is."""
        result = _find_ai_integration_id(
            ai_integrations_api, ai_integration_info["id"], None
        )
        assert result == ai_integration_info["id"]


# ---------------------------------------------------------------------------
# _find_role_id
# ---------------------------------------------------------------------------
class TestFindRoleId:
    """Tests for _find_role_id."""

    @pytest.fixture(scope="class")
    def roles_api(self, generated_client) -> Any:
        from arize._generated import api_client as gen

        return gen.RolesApi(generated_client)

    @pytest.fixture(scope="class")
    def role_info(self, generated_client) -> dict[str, Any]:
        """Fetch a real role (any type) from the account."""
        from arize._generated import api_client as gen

        api = gen.RolesApi(generated_client)
        resp = api.roles_list(limit=1)
        if not resp.roles:
            pytest.skip("No roles in account")
        r = resp.roles[0]
        return {"id": r.id, "name": r.name}

    def test_resolve_by_name(
        self, roles_api, role_info: dict[str, Any]
    ) -> None:
        """Resolves a role name to its ID."""
        result = _find_role_id(roles_api, role_info["name"])
        assert result == role_info["id"]

    def test_id_passthrough(self, roles_api, role_info: dict[str, Any]) -> None:
        """A base64 ID is returned as-is without calling roles_list."""
        result = _find_role_id(roles_api, role_info["id"])
        assert result == role_info["id"]

    def test_not_found_raises(self, roles_api) -> None:
        """Raises NotFoundError for an unknown role name."""
        with pytest.raises(NotFoundError, match="role"):
            _find_role_id(roles_api, "nonexistent-role-abc-xyz-12345")
