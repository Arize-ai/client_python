"""Unit tests for arize.utils.resolve."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from arize.exceptions.spaces import AmbiguousNameError
from arize.utils.resolve import (
    NotFoundError,
    _find_ai_integration_id,
    _find_annotation_config_id,
    _find_dataset_id,
    _find_evaluator_id,
    _find_experiment_id,
    _find_integration_id,
    _find_project_id,
    _find_prompt_id,
    _find_space_id,
    _find_task_id,
    _resolve_resource,
    is_resource_id,
)

# A valid base64 identifier (decodes to "Space:9050:1JkR")
B64_ID = "U3BhY2U6OTA1MDoxSmtS"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paginated(items: list, next_cursor: str | None = None) -> MagicMock:
    """Build a mock paginated response with .pagination.next_cursor."""
    resp = MagicMock()
    resp.pagination.next_cursor = next_cursor
    return resp


def _item(
    name: str, id: str = "some-id", dataset_id: str | None = None
) -> MagicMock:
    item = MagicMock()
    item.name = name
    item.id = id
    item.dataset_id = dataset_id
    return item


# ---------------------------------------------------------------------------
# AmbiguousNameError
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAmbiguousNameError:
    def test_message_contains_resource_type_name_and_ids(self) -> None:
        err = AmbiguousNameError("space", "my-space", ["id-1", "id-2"])
        msg = str(err)
        assert "space" in msg
        assert "my-space" in msg
        assert "id-1" in msg
        assert "id-2" in msg

    def test_attributes(self) -> None:
        err = AmbiguousNameError("space", "my-space", ["id-1", "id-2"])
        assert err.resource_type == "space"
        assert err.resource_name == "my-space"
        assert err.matching_ids == ["id-1", "id-2"]

    def test_message_suggests_id(self) -> None:
        err = AmbiguousNameError("space", "my-space", ["id-1", "id-2"])
        assert "ID" in str(err)


# ---------------------------------------------------------------------------
# NotFoundError
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNotFoundError:
    def test_message_without_available_or_hint(self) -> None:
        err = NotFoundError("space", "my-space")
        assert "space 'my-space' not found" in str(err)
        assert err.resource_type == "space"
        assert err.resource_name == "my-space"
        assert err.available_names == []

    def test_message_with_available_names(self) -> None:
        err = NotFoundError("space", "x", available_names=["a", "b"])
        assert "Available spaces: a, b" in str(err)
        assert err.available_names == ["a", "b"]

    def test_message_with_hint(self) -> None:
        err = NotFoundError("project", "x", hint="Try providing a space.")
        assert "Try providing a space." in str(err)

    def test_empty_available_names_not_shown(self) -> None:
        err = NotFoundError("space", "x", available_names=[])
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
        mock_api.list_spaces.return_value = resp
        result = _find_space_id(mock_api, "my-space")
        assert result == "found-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.spaces = [_item("other-space", "other-id")]
        mock_api = MagicMock()
        mock_api.list_spaces.return_value = resp
        with pytest.raises(NotFoundError, match="space"):
            _find_space_id(mock_api, "my-space")

    def test_pagination_fetches_next_page(self) -> None:
        page1 = _make_paginated([], next_cursor="cursor-abc")
        page1.spaces = [_item("other-space")]
        page2 = _make_paginated([])
        page2.spaces = [_item("my-space", "found-id")]
        mock_api = MagicMock()
        mock_api.list_spaces.side_effect = [page1, page2]
        result = _find_space_id(mock_api, "my-space")
        assert result == "found-id"
        assert mock_api.list_spaces.call_count == 2

    def test_duplicate_name_raises_ambiguous_error(self) -> None:
        resp = _make_paginated([])
        resp.spaces = [
            _item("my-space", "id-org-a"),
            _item("my-space", "id-org-b"),
        ]
        mock_api = MagicMock()
        mock_api.list_spaces.return_value = resp
        with pytest.raises(AmbiguousNameError) as exc_info:
            _find_space_id(mock_api, "my-space")
        err = exc_info.value
        assert err.resource_name == "my-space"
        assert set(err.matching_ids) == {"id-org-a", "id-org-b"}

    def test_duplicate_name_across_pages_raises_ambiguous_error(self) -> None:
        page1 = _make_paginated([], next_cursor="cursor-abc")
        page1.spaces = [_item("my-space", "id-org-a")]
        page2 = _make_paginated([])
        page2.spaces = [_item("my-space", "id-org-b")]
        mock_api = MagicMock()
        mock_api.list_spaces.side_effect = [page1, page2]
        with pytest.raises(AmbiguousNameError) as exc_info:
            _find_space_id(mock_api, "my-space")
        err = exc_info.value
        assert set(err.matching_ids) == {"id-org-a", "id-org-b"}


# ---------------------------------------------------------------------------
# _find_project_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindProjectId:
    def test_base64_passthrough(self) -> None:
        assert (
            _find_project_id(MagicMock(), MagicMock(), B64_ID, None) == B64_ID
        )

    def test_no_space_raises(self) -> None:
        with pytest.raises(NotFoundError, match="project"):
            _find_project_id(MagicMock(), MagicMock(), "my-project", None)

    def test_name_resolved_with_space_id(self) -> None:
        resp = _make_paginated([])
        resp.projects = [_item("my-project", "proj-id")]
        mock_api = MagicMock()
        mock_api.list_projects.return_value = resp
        result = _find_project_id(mock_api, MagicMock(), "my-project", B64_ID)
        assert result == "proj-id"

    def test_name_resolved_with_space_name(self) -> None:
        resp = _make_paginated([])
        resp.projects = [_item("my-project", "proj-id")]
        mock_api = MagicMock()
        mock_api.list_projects.return_value = resp
        mock_spaces_api = MagicMock()
        mock_spaces_api.list_spaces.return_value = _make_paginated([])
        mock_spaces_api.list_spaces.return_value.spaces = [
            _item("sname", B64_ID)
        ]
        result = _find_project_id(
            mock_api, mock_spaces_api, "my-project", "sname"
        )
        assert result == "proj-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.projects = [_item("other-project")]
        mock_api = MagicMock()
        mock_api.list_projects.return_value = resp
        with pytest.raises(NotFoundError, match="project"):
            _find_project_id(mock_api, MagicMock(), "missing", B64_ID)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.projects = [_item("other")]
        page2 = _make_paginated([])
        page2.projects = [_item("my-project", "proj-id")]
        mock_api = MagicMock()
        mock_api.list_projects.side_effect = [page1, page2]
        assert (
            _find_project_id(mock_api, MagicMock(), "my-project", B64_ID)
            == "proj-id"
        )


# ---------------------------------------------------------------------------
# _find_dataset_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindDatasetId:
    def test_base64_passthrough(self) -> None:
        assert (
            _find_dataset_id(MagicMock(), MagicMock(), B64_ID, None) == B64_ID
        )

    def test_no_space_raises(self) -> None:
        with pytest.raises(NotFoundError, match="dataset"):
            _find_dataset_id(MagicMock(), MagicMock(), "my-dataset", None)

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.datasets = [_item("my-dataset", "ds-id")]
        mock_api = MagicMock()
        mock_api.list_datasets.return_value = resp
        result = _find_dataset_id(mock_api, MagicMock(), "my-dataset", B64_ID)
        assert result == "ds-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.datasets = [_item("other-dataset")]
        mock_api = MagicMock()
        mock_api.list_datasets.return_value = resp
        with pytest.raises(NotFoundError, match="dataset"):
            _find_dataset_id(mock_api, MagicMock(), "missing", B64_ID)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.datasets = [_item("other")]
        page2 = _make_paginated([])
        page2.datasets = [_item("my-dataset", "ds-id")]
        mock_api = MagicMock()
        mock_api.list_datasets.side_effect = [page1, page2]
        assert (
            _find_dataset_id(mock_api, MagicMock(), "my-dataset", B64_ID)
            == "ds-id"
        )


# ---------------------------------------------------------------------------
# _find_experiment_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindExperimentId:
    def test_base64_passthrough(self) -> None:
        assert (
            _find_experiment_id(
                MagicMock(), MagicMock(), MagicMock(), B64_ID, None, None
            )
            == B64_ID
        )

    def test_no_dataset_or_space_raises(self) -> None:
        with pytest.raises(NotFoundError, match="experiment"):
            _find_experiment_id(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                "my-experiment",
                None,
                None,
            )

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.experiments = [_item("my-experiment", "exp-id")]
        mock_api = MagicMock()
        mock_api.list_experiments.return_value = resp
        # Use B64_ID as dataset so _find_dataset_id is skipped (direct ID passthrough)
        result = _find_experiment_id(
            mock_api, MagicMock(), MagicMock(), "my-experiment", B64_ID, None
        )
        assert result == "exp-id"

    def test_name_resolved_standalone_via_space(self) -> None:
        """No dataset provided: resolves a standalone experiment by name
        within the space instead.
        """
        resp = _make_paginated([])
        resp.experiments = [_item("my-experiment", "exp-id")]
        mock_api = MagicMock()
        mock_api.list_experiments.return_value = resp
        # Use B64_ID as space so _find_space_id is skipped (direct ID passthrough)
        result = _find_experiment_id(
            mock_api, MagicMock(), MagicMock(), "my-experiment", None, B64_ID
        )
        assert result == "exp-id"
        mock_api.list_experiments.assert_called_once()
        assert mock_api.list_experiments.call_args.kwargs["space_id"] == B64_ID

    def test_name_resolved_via_space_when_match_is_dataset_backed(self) -> None:
        """No dataset provided, and the only name match in the space is a
        dataset-associated experiment (not standalone): still resolves it,
        since there's no collision to disambiguate.
        """
        resp = _make_paginated([])
        resp.experiments = [_item("my-experiment", "exp-id", dataset_id="ds-1")]
        mock_api = MagicMock()
        mock_api.list_experiments.return_value = resp
        result = _find_experiment_id(
            mock_api, MagicMock(), MagicMock(), "my-experiment", None, B64_ID
        )
        assert result == "exp-id"

    def test_name_ambiguous_across_standalone_and_dataset_backed_raises(
        self,
    ) -> None:
        """No dataset provided, and the name matches both a standalone and a
        dataset-associated experiment in the space: raises rather than
        silently returning either one.
        """
        resp = _make_paginated([])
        resp.experiments = [
            _item("my-experiment", "standalone-id", dataset_id=None),
            _item("my-experiment", "dataset-backed-id", dataset_id="ds-1"),
        ]
        mock_api = MagicMock()
        mock_api.list_experiments.return_value = resp
        with pytest.raises(AmbiguousNameError, match="my-experiment"):
            _find_experiment_id(
                mock_api,
                MagicMock(),
                MagicMock(),
                "my-experiment",
                None,
                B64_ID,
            )

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.experiments = [_item("other-experiment")]
        mock_api = MagicMock()
        mock_api.list_experiments.return_value = resp
        with pytest.raises(NotFoundError, match="experiment"):
            _find_experiment_id(
                mock_api, MagicMock(), MagicMock(), "missing", B64_ID, None
            )

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.experiments = [_item("other")]
        page2 = _make_paginated([])
        page2.experiments = [_item("my-exp", "exp-id")]
        mock_api = MagicMock()
        mock_api.list_experiments.side_effect = [page1, page2]
        assert (
            _find_experiment_id(
                mock_api, MagicMock(), MagicMock(), "my-exp", B64_ID, None
            )
            == "exp-id"
        )


# ---------------------------------------------------------------------------
# _find_prompt_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindPromptId:
    def test_base64_passthrough(self) -> None:
        assert _find_prompt_id(MagicMock(), MagicMock(), B64_ID, None) == B64_ID

    def test_no_space_raises(self) -> None:
        with pytest.raises(NotFoundError, match="prompt"):
            _find_prompt_id(MagicMock(), MagicMock(), "my-prompt", None)

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.prompts = [_item("my-prompt", "pr-id")]
        mock_api = MagicMock()
        mock_api.list_prompts.return_value = resp
        result = _find_prompt_id(mock_api, MagicMock(), "my-prompt", B64_ID)
        assert result == "pr-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.prompts = [_item("other-prompt")]
        mock_api = MagicMock()
        mock_api.list_prompts.return_value = resp
        with pytest.raises(NotFoundError, match="prompt"):
            _find_prompt_id(mock_api, MagicMock(), "missing", B64_ID)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.prompts = [_item("other")]
        page2 = _make_paginated([])
        page2.prompts = [_item("my-prompt", "pr-id")]
        mock_api = MagicMock()
        mock_api.list_prompts.side_effect = [page1, page2]
        assert (
            _find_prompt_id(mock_api, MagicMock(), "my-prompt", B64_ID)
            == "pr-id"
        )


# ---------------------------------------------------------------------------
# _find_evaluator_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindEvaluatorId:
    def test_base64_passthrough(self) -> None:
        assert (
            _find_evaluator_id(MagicMock(), MagicMock(), B64_ID, None) == B64_ID
        )

    def test_no_space_raises(self) -> None:
        with pytest.raises(NotFoundError, match="evaluator"):
            _find_evaluator_id(MagicMock(), MagicMock(), "my-evaluator", None)

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.evaluators = [_item("my-evaluator", "ev-id")]
        mock_api = MagicMock()
        mock_api.list_evaluators.return_value = resp
        result = _find_evaluator_id(
            mock_api, MagicMock(), "my-evaluator", B64_ID
        )
        assert result == "ev-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.evaluators = [_item("other-evaluator")]
        mock_api = MagicMock()
        mock_api.list_evaluators.return_value = resp
        with pytest.raises(NotFoundError, match="evaluator"):
            _find_evaluator_id(mock_api, MagicMock(), "missing", B64_ID)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.evaluators = [_item("other")]
        page2 = _make_paginated([])
        page2.evaluators = [_item("my-evaluator", "ev-id")]
        mock_api = MagicMock()
        mock_api.list_evaluators.side_effect = [page1, page2]
        assert (
            _find_evaluator_id(mock_api, MagicMock(), "my-evaluator", B64_ID)
            == "ev-id"
        )


# ---------------------------------------------------------------------------
# _find_annotation_config_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindAnnotationConfigId:
    def test_base64_passthrough(self) -> None:
        assert (
            _find_annotation_config_id(MagicMock(), MagicMock(), B64_ID, None)
            == B64_ID
        )

    def test_no_space_raises(self) -> None:
        with pytest.raises(NotFoundError, match="annotation config"):
            _find_annotation_config_id(
                MagicMock(), MagicMock(), "my-config", None
            )

    def test_name_resolved(self) -> None:
        inner = MagicMock()
        inner.name = "my-config"
        inner.id = "ac-id"
        ac = MagicMock()
        ac.actual_instance = inner

        resp = _make_paginated([])
        resp.annotation_configs = [ac]
        mock_api = MagicMock()
        mock_api.list_annotation_configs.return_value = resp
        result = _find_annotation_config_id(
            mock_api, MagicMock(), "my-config", B64_ID
        )
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
        mock_api.list_annotation_configs.return_value = resp
        result = _find_annotation_config_id(
            mock_api, MagicMock(), "my-config", B64_ID
        )
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
        mock_api.list_annotation_configs.return_value = resp
        with pytest.raises(NotFoundError, match="annotation config"):
            _find_annotation_config_id(mock_api, MagicMock(), "missing", B64_ID)

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
        mock_api.list_annotation_configs.side_effect = [page1, page2]
        assert (
            _find_annotation_config_id(
                mock_api, MagicMock(), "my-config", B64_ID
            )
            == "ac-id"
        )


# ---------------------------------------------------------------------------
# _find_ai_integration_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindAiIntegrationId:
    def test_base64_passthrough(self) -> None:
        assert (
            _find_ai_integration_id(MagicMock(), MagicMock(), B64_ID, None)
            == B64_ID
        )

    def test_no_space_raises(self) -> None:
        with pytest.raises(NotFoundError, match="AI integration"):
            _find_ai_integration_id(
                MagicMock(), MagicMock(), "my-integration", None
            )

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.ai_integrations = [_item("my-integration", "ai-id")]
        mock_api = MagicMock()
        mock_api.list_ai_integrations.return_value = resp
        result = _find_ai_integration_id(
            mock_api, MagicMock(), "my-integration", B64_ID
        )
        assert result == "ai-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.ai_integrations = [_item("other-integration")]
        mock_api = MagicMock()
        mock_api.list_ai_integrations.return_value = resp
        with pytest.raises(NotFoundError, match="AI integration"):
            _find_ai_integration_id(mock_api, MagicMock(), "missing", B64_ID)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.ai_integrations = [_item("other")]
        page2 = _make_paginated([])
        page2.ai_integrations = [_item("my-integration", "ai-id")]
        mock_api = MagicMock()
        mock_api.list_ai_integrations.side_effect = [page1, page2]
        assert (
            _find_ai_integration_id(
                mock_api, MagicMock(), "my-integration", B64_ID
            )
            == "ai-id"
        )


# ---------------------------------------------------------------------------
# _find_integration_id
# ---------------------------------------------------------------------------


def _wrapped_item(name: str, id: str = "some-id") -> MagicMock:
    """Build a mock Integration oneOf wrapper with an inner actual_instance."""
    inner = MagicMock()
    inner.name = name
    inner.id = id
    wrapper = MagicMock()
    wrapper.actual_instance = inner
    return wrapper


@pytest.mark.unit
class TestFindIntegrationId:
    def test_base64_passthrough(self) -> None:
        assert _find_integration_id(MagicMock(), B64_ID, "LLM", None) == B64_ID

    def test_base64_passthrough_without_type(self) -> None:
        """An ID needs no type — it identifies the integration on its own."""
        mock_api = MagicMock()
        assert _find_integration_id(mock_api, B64_ID, None, None) == B64_ID
        mock_api.list_integrations.assert_not_called()

    def test_name_without_type_raises(self) -> None:
        """A name is only unique per (account, type), so type is required."""
        mock_api = MagicMock()
        with pytest.raises(NotFoundError, match="integration_type"):
            _find_integration_id(mock_api, "my-integration", None, None)
        mock_api.list_integrations.assert_not_called()

    def test_no_space_resolves_by_type(self) -> None:
        """Without a space, resolution still lists by type and raises only
        when the name is not found.

        Integration names are unique per ``(account, type)``, so ``space`` is
        an optional visibility filter, not required to resolve a name. This
        also guards against the pagination loop never terminating: a real
        exhausted response has ``next_cursor=None``.
        """
        resp = _make_paginated([])
        resp.integrations = []
        mock_api = MagicMock()
        mock_api.list_integrations.return_value = resp
        with pytest.raises(NotFoundError, match="integration"):
            _find_integration_id(mock_api, "my-integration", "LLM", None)
        # space is optional: the lookup proceeds with no space filter.
        kwargs = mock_api.list_integrations.call_args.kwargs
        assert kwargs["space_id"] is None
        assert kwargs["space_name"] is None

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.integrations = [_wrapped_item("my-integration", "int-id")]
        mock_api = MagicMock()
        mock_api.list_integrations.return_value = resp
        result = _find_integration_id(
            mock_api, "my-integration", "AGENT", B64_ID
        )
        assert result == "int-id"
        # type should be forwarded to the list endpoint
        assert mock_api.list_integrations.call_args.kwargs["type"] == "AGENT"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.integrations = [_wrapped_item("other-integration")]
        mock_api = MagicMock()
        mock_api.list_integrations.return_value = resp
        with pytest.raises(NotFoundError, match="integration"):
            _find_integration_id(mock_api, "missing", "LLM", B64_ID)

    def test_none_actual_instance_skipped(self) -> None:
        empty = MagicMock()
        empty.actual_instance = None
        resp = _make_paginated([])
        resp.integrations = [empty, _wrapped_item("my-integration", "int-id")]
        mock_api = MagicMock()
        mock_api.list_integrations.return_value = resp
        assert (
            _find_integration_id(mock_api, "my-integration", "LLM", B64_ID)
            == "int-id"
        )

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.integrations = [_wrapped_item("other")]
        page2 = _make_paginated([])
        page2.integrations = [_wrapped_item("my-integration", "int-id")]
        mock_api = MagicMock()
        mock_api.list_integrations.side_effect = [page1, page2]
        assert (
            _find_integration_id(mock_api, "my-integration", "LLM", B64_ID)
            == "int-id"
        )


# ---------------------------------------------------------------------------
# _find_task_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindTaskId:
    def test_base64_passthrough(self) -> None:
        assert _find_task_id(MagicMock(), MagicMock(), B64_ID, None) == B64_ID

    def test_no_space_raises(self) -> None:
        with pytest.raises(NotFoundError, match="task"):
            _find_task_id(MagicMock(), MagicMock(), "my-task", None)

    def test_name_resolved(self) -> None:
        resp = _make_paginated([])
        resp.tasks = [_item("my-task", "task-id")]
        mock_api = MagicMock()
        mock_api.list_tasks.return_value = resp
        result = _find_task_id(mock_api, MagicMock(), "my-task", B64_ID)
        assert result == "task-id"

    def test_name_not_found_raises(self) -> None:
        resp = _make_paginated([])
        resp.tasks = [_item("other-task")]
        mock_api = MagicMock()
        mock_api.list_tasks.return_value = resp
        with pytest.raises(NotFoundError, match="task"):
            _find_task_id(mock_api, MagicMock(), "missing", B64_ID)

    def test_pagination(self) -> None:
        page1 = _make_paginated([], next_cursor="c")
        page1.tasks = [_item("other")]
        page2 = _make_paginated([])
        page2.tasks = [_item("my-task", "task-id")]
        mock_api = MagicMock()
        mock_api.list_tasks.side_effect = [page1, page2]
        assert (
            _find_task_id(mock_api, MagicMock(), "my-task", B64_ID) == "task-id"
        )
