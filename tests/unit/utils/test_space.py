"""Unit tests for arize.utils.resolve."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from arize.utils.resolve import (
    ResolutionError,
    _find_ai_integration_id,
    _find_annotation_config_id,
    _find_dataset_id,
    _find_evaluator_id,
    _find_experiment_id,
    _find_project_id,
    _find_prompt_id,
    _find_space_id,
    _find_task_id,
    _resolve_resource,
    is_resource_id,
)

# A valid base64 global ID (decodes to "Space:9050:1JkR")
B64_ID = "U3BhY2U6OTA1MDoxSmtS"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paginated(items: list, next_cursor: str | None = None) -> MagicMock:
    """Build a mock paginated response with .pagination.next_cursor."""
    resp = MagicMock()
    resp.pagination.next_cursor = next_cursor
    return resp


def _item(name: str, id: str = "some-id") -> MagicMock:
    item = MagicMock()
    item.name = name
    item.id = id
    return item


# ---------------------------------------------------------------------------
# ResolutionError
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolutionError:
    def test_message_without_available_or_hint(self) -> None:
        err = ResolutionError("space", "my-space")
        assert "space 'my-space' not found" in str(err)
        assert err.resource_type == "space"
        assert err.resource_name == "my-space"
        assert err.available_names == []

    def test_message_with_available_names(self) -> None:
        err = ResolutionError("space", "x", available_names=["a", "b"])
        assert "Available spaces: a, b" in str(err)
        assert err.available_names == ["a", "b"]

    def test_message_with_hint(self) -> None:
        err = ResolutionError("project", "x", hint="Try providing a space.")
        assert "Try providing a space." in str(err)

    def test_empty_available_names_not_shown(self) -> None:
        err = ResolutionError("space", "x", available_names=[])
        assert "Available" not in str(err)


# ---------------------------------------------------------------------------
# _resolve_resource / is_resource_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveResource:
    def test_none_returns_none_pair(self) -> None:
        r = _resolve_resource(None)
        assert r.id is None
        assert r.name is None

    def test_plain_string_returns_space_name(self) -> None:
        r = _resolve_resource("my-space")
        assert r.id is None
        assert r.name == "my-space"

    def test_base64_global_id_returns_space_id(self) -> None:
        r = _resolve_resource(B64_ID)
        assert r.id == B64_ID
        assert r.name is None

    def test_empty_string_returns_name(self) -> None:
        r = _resolve_resource("")
        assert r.id is None
        assert r.name == ""

    def test_is_id_method(self) -> None:
        assert _resolve_resource(B64_ID).is_id()
        assert not _resolve_resource("my-space").is_id()

    def test_is_name_method(self) -> None:
        assert _resolve_resource("my-space").is_name()
        assert not _resolve_resource(B64_ID).is_name()


@pytest.mark.unit
class TestIsResourceId:
    def test_base64_id_returns_true(self) -> None:
        assert is_resource_id(B64_ID)

    def test_plain_name_returns_false(self) -> None:
        assert not is_resource_id("my-space")

    def test_empty_string_returns_false(self) -> None:
        assert not is_resource_id("")


# ---------------------------------------------------------------------------
# _find_space_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindSpaceId:
    def test_base64_passthrough_skips_api(self) -> None:
        mock_api = MagicMock()
        assert _find_space_id(mock_api, B64_ID) == B64_ID
        mock_api.assert_not_called()

    def test_name_resolved_to_id(self) -> None:
        resp = _make_paginated([])
        resp.spaces = [_item("my-space", "found-id")]
        mock_api = MagicMock()
        mock_api.spaces_list.return_value = resp
        result = _find_space_id(mock_api, "my-space")
        assert result == "found-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.spaces = [_item("other-space", "other-id")]
        mock_api = MagicMock()
        mock_api.spaces_list.return_value = resp
        with pytest.raises(ResolutionError, match="space"):
            _find_space_id(mock_api, "my-space")

    def test_pagination_fetches_next_page(self) -> None:
        page1 = _make_paginated([], next_cursor="cursor-abc")
        page1.spaces = [_item("other-space")]
        page2 = _make_paginated([])
        page2.spaces = [_item("my-space", "found-id")]
        mock_api = MagicMock()
        mock_api.spaces_list.side_effect = [page1, page2]
        result = _find_space_id(mock_api, "my-space")
        assert result == "found-id"
        assert mock_api.spaces_list.call_count == 2


# ---------------------------------------------------------------------------
# _find_project_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindProjectId:
    def test_base64_passthrough(self) -> None:
        assert _find_project_id(MagicMock(), B64_ID, None) == B64_ID

    def test_no_space_raises(self) -> None:
        with pytest.raises(ResolutionError, match="project"):
            _find_project_id(MagicMock(), "my-project", None)

    def test_name_resolved_with_space_id(self) -> None:
        resp = _make_paginated([])
        resp.projects = [_item("my-project", "proj-id")]
        mock_api = MagicMock()
        mock_api.projects_list.return_value = resp
        result = _find_project_id(mock_api, "my-project", B64_ID)
        assert result == "proj-id"

    def test_name_resolved_with_space_name(self) -> None:
        resp = _make_paginated([])
        resp.projects = [_item("my-project", "proj-id")]
        mock_api = MagicMock()
        mock_api.projects_list.return_value = resp
        result = _find_project_id(mock_api, "my-project", "sname")
        assert result == "proj-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.projects = [_item("other-project")]
        mock_api = MagicMock()
        mock_api.projects_list.return_value = resp
        with pytest.raises(ResolutionError, match="project"):
            _find_project_id(mock_api, "missing", B64_ID)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.projects = [_item("other")]
        page2 = _make_paginated([])
        page2.projects = [_item("my-project", "proj-id")]
        mock_api = MagicMock()
        mock_api.projects_list.side_effect = [page1, page2]
        assert _find_project_id(mock_api, "my-project", B64_ID) == "proj-id"


# ---------------------------------------------------------------------------
# _find_dataset_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindDatasetId:
    def test_base64_passthrough(self) -> None:
        assert _find_dataset_id(MagicMock(), B64_ID, None) == B64_ID

    def test_no_space_raises(self) -> None:
        with pytest.raises(ResolutionError, match="dataset"):
            _find_dataset_id(MagicMock(), "my-dataset", None)

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.datasets = [_item("my-dataset", "ds-id")]
        mock_api = MagicMock()
        mock_api.datasets_list.return_value = resp
        result = _find_dataset_id(mock_api, "my-dataset", B64_ID)
        assert result == "ds-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.datasets = [_item("other-dataset")]
        mock_api = MagicMock()
        mock_api.datasets_list.return_value = resp
        with pytest.raises(ResolutionError, match="dataset"):
            _find_dataset_id(mock_api, "missing", B64_ID)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.datasets = [_item("other")]
        page2 = _make_paginated([])
        page2.datasets = [_item("my-dataset", "ds-id")]
        mock_api = MagicMock()
        mock_api.datasets_list.side_effect = [page1, page2]
        assert _find_dataset_id(mock_api, "my-dataset", B64_ID) == "ds-id"


# ---------------------------------------------------------------------------
# _find_experiment_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindExperimentId:
    def test_base64_passthrough(self) -> None:
        assert (
            _find_experiment_id(MagicMock(), MagicMock(), B64_ID, None, None)
            == B64_ID
        )

    def test_no_dataset_id_raises(self) -> None:
        with pytest.raises(ResolutionError, match="experiment"):
            _find_experiment_id(
                MagicMock(), MagicMock(), "my-experiment", None, None
            )

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.experiments = [_item("my-experiment", "exp-id")]
        mock_api = MagicMock()
        mock_api.experiments_list.return_value = resp
        # Use B64_ID as dataset so _find_dataset_id is skipped (direct ID passthrough)
        result = _find_experiment_id(
            mock_api, MagicMock(), "my-experiment", B64_ID, None
        )
        assert result == "exp-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.experiments = [_item("other-experiment")]
        mock_api = MagicMock()
        mock_api.experiments_list.return_value = resp
        with pytest.raises(ResolutionError, match="experiment"):
            _find_experiment_id(mock_api, MagicMock(), "missing", B64_ID, None)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.experiments = [_item("other")]
        page2 = _make_paginated([])
        page2.experiments = [_item("my-exp", "exp-id")]
        mock_api = MagicMock()
        mock_api.experiments_list.side_effect = [page1, page2]
        assert (
            _find_experiment_id(mock_api, MagicMock(), "my-exp", B64_ID, None)
            == "exp-id"
        )


# ---------------------------------------------------------------------------
# _find_prompt_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindPromptId:
    def test_base64_passthrough(self) -> None:
        assert _find_prompt_id(MagicMock(), B64_ID, None) == B64_ID

    def test_no_space_raises(self) -> None:
        with pytest.raises(ResolutionError, match="prompt"):
            _find_prompt_id(MagicMock(), "my-prompt", None)

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.prompts = [_item("my-prompt", "pr-id")]
        mock_api = MagicMock()
        mock_api.prompts_list.return_value = resp
        result = _find_prompt_id(mock_api, "my-prompt", B64_ID)
        assert result == "pr-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.prompts = [_item("other-prompt")]
        mock_api = MagicMock()
        mock_api.prompts_list.return_value = resp
        with pytest.raises(ResolutionError, match="prompt"):
            _find_prompt_id(mock_api, "missing", B64_ID)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.prompts = [_item("other")]
        page2 = _make_paginated([])
        page2.prompts = [_item("my-prompt", "pr-id")]
        mock_api = MagicMock()
        mock_api.prompts_list.side_effect = [page1, page2]
        assert _find_prompt_id(mock_api, "my-prompt", B64_ID) == "pr-id"


# ---------------------------------------------------------------------------
# _find_evaluator_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindEvaluatorId:
    def test_base64_passthrough(self) -> None:
        assert _find_evaluator_id(MagicMock(), B64_ID, None) == B64_ID

    def test_no_space_raises(self) -> None:
        with pytest.raises(ResolutionError, match="evaluator"):
            _find_evaluator_id(MagicMock(), "my-evaluator", None)

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.evaluators = [_item("my-evaluator", "ev-id")]
        mock_api = MagicMock()
        mock_api.evaluators_list.return_value = resp
        result = _find_evaluator_id(mock_api, "my-evaluator", B64_ID)
        assert result == "ev-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.evaluators = [_item("other-evaluator")]
        mock_api = MagicMock()
        mock_api.evaluators_list.return_value = resp
        with pytest.raises(ResolutionError, match="evaluator"):
            _find_evaluator_id(mock_api, "missing", B64_ID)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.evaluators = [_item("other")]
        page2 = _make_paginated([])
        page2.evaluators = [_item("my-evaluator", "ev-id")]
        mock_api = MagicMock()
        mock_api.evaluators_list.side_effect = [page1, page2]
        assert _find_evaluator_id(mock_api, "my-evaluator", B64_ID) == "ev-id"


# ---------------------------------------------------------------------------
# _find_annotation_config_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindAnnotationConfigId:
    def test_base64_passthrough(self) -> None:
        assert _find_annotation_config_id(MagicMock(), B64_ID, None) == B64_ID

    def test_no_space_raises(self) -> None:
        with pytest.raises(ResolutionError, match="annotation config"):
            _find_annotation_config_id(MagicMock(), "my-config", None)

    def test_name_resolved(self) -> None:
        inner = MagicMock()
        inner.name = "my-config"
        inner.id = "ac-id"
        ac = MagicMock()
        ac.actual_instance = inner

        resp = _make_paginated([])
        resp.annotation_configs = [ac]
        mock_api = MagicMock()
        mock_api.annotation_configs_list.return_value = resp
        result = _find_annotation_config_id(mock_api, "my-config", B64_ID)
        assert result == "ac-id"

    def test_skips_none_inner_instance(self) -> None:
        ac_null = MagicMock()
        ac_null.actual_instance = None
        inner = MagicMock()
        inner.name = "my-config"
        inner.id = "ac-id"
        ac_real = MagicMock()
        ac_real.actual_instance = inner

        resp = _make_paginated([])
        resp.annotation_configs = [ac_null, ac_real]
        mock_api = MagicMock()
        mock_api.annotation_configs_list.return_value = resp
        result = _find_annotation_config_id(mock_api, "my-config", B64_ID)
        assert result == "ac-id"

    def test_name_not_found_raises(self) -> None:
        inner = MagicMock()
        inner.name = "other-config"
        inner.id = "other-id"
        ac = MagicMock()
        ac.actual_instance = inner
        resp = _make_paginated([])
        resp.annotation_configs = [ac]
        mock_api = MagicMock()
        mock_api.annotation_configs_list.return_value = resp
        with pytest.raises(ResolutionError, match="annotation config"):
            _find_annotation_config_id(mock_api, "missing", B64_ID)

    def test_pagination(self) -> None:
        inner1 = MagicMock()
        inner1.name = "other"
        inner1.id = "x"
        ac1 = MagicMock()
        ac1.actual_instance = inner1
        inner2 = MagicMock()
        inner2.name = "my-config"
        inner2.id = "ac-id"
        ac2 = MagicMock()
        ac2.actual_instance = inner2
        page1 = _make_paginated([], next_cursor="c")
        page1.annotation_configs = [ac1]
        page2 = _make_paginated([])
        page2.annotation_configs = [ac2]
        mock_api = MagicMock()
        mock_api.annotation_configs_list.side_effect = [page1, page2]
        assert (
            _find_annotation_config_id(mock_api, "my-config", B64_ID) == "ac-id"
        )


# ---------------------------------------------------------------------------
# _find_ai_integration_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindAiIntegrationId:
    def test_base64_passthrough(self) -> None:
        assert _find_ai_integration_id(MagicMock(), B64_ID, None) == B64_ID

    def test_no_space_raises(self) -> None:
        with pytest.raises(ResolutionError, match="AI integration"):
            _find_ai_integration_id(MagicMock(), "my-integration", None)

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.ai_integrations = [_item("my-integration", "ai-id")]
        mock_api = MagicMock()
        mock_api.ai_integrations_list.return_value = resp
        result = _find_ai_integration_id(mock_api, "my-integration", B64_ID)
        assert result == "ai-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.ai_integrations = [_item("other-integration")]
        mock_api = MagicMock()
        mock_api.ai_integrations_list.return_value = resp
        with pytest.raises(ResolutionError, match="AI integration"):
            _find_ai_integration_id(mock_api, "missing", B64_ID)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.ai_integrations = [_item("other")]
        page2 = _make_paginated([])
        page2.ai_integrations = [_item("my-integration", "ai-id")]
        mock_api = MagicMock()
        mock_api.ai_integrations_list.side_effect = [page1, page2]
        assert (
            _find_ai_integration_id(mock_api, "my-integration", B64_ID)
            == "ai-id"
        )


# ---------------------------------------------------------------------------
# _find_task_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindTaskId:
    def test_base64_passthrough(self) -> None:
        assert _find_task_id(MagicMock(), B64_ID, None) == B64_ID

    def test_no_space_raises(self) -> None:
        with pytest.raises(ResolutionError, match="task"):
            _find_task_id(MagicMock(), "my-task", None)

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.tasks = [_item("my-task", "task-id")]
        mock_api = MagicMock()
        mock_api.tasks_list.return_value = resp
        result = _find_task_id(mock_api, "my-task", B64_ID)
        assert result == "task-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.tasks = [_item("other-task")]
        mock_api = MagicMock()
        mock_api.tasks_list.return_value = resp
        with pytest.raises(ResolutionError, match="task"):
            _find_task_id(mock_api, "missing", B64_ID)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.tasks = [_item("other")]
        page2 = _make_paginated([])
        page2.tasks = [_item("my-task", "task-id")]
        mock_api = MagicMock()
        mock_api.tasks_list.side_effect = [page1, page2]
        assert _find_task_id(mock_api, "my-task", B64_ID) == "task-id"
