"""Unit tests for the space-name-to-ID fix in arize.utils.resolve."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from arize.exceptions.spaces import AmbiguousNameError
from arize.utils.resolve import (
    NotFoundError,
    _find_dataset_id,
    _find_project_id,
    _find_webhook_id,
)

# A valid base64 identifier (decodes to "Space:9050:1JkR")
_SPACE_ID = "U3BhY2U6OTA1MDoxSmtS"

# A valid base64 identifier (decodes to "Project:123")
_PROJECT_ID = "UHJvamVjdDoxMjM="

# A valid base64 identifier (decodes to "Dataset:123")
_DATASET_ID = "RGF0YXNldDoxMjM="

# A valid base64 identifier (decodes to "Webhook:123")
_WEBHOOK_ID = "V2ViaG9vazoxMjM="

# A valid base64 identifier (decodes to "Organization:123")
_ORG_ID = "T3JnYW5pemF0aW9uOjEyMw=="


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spaces_api(space_name: str, space_id: str) -> MagicMock:
    """Return a SpacesApi mock that resolves *space_name* to *space_id*."""
    space = MagicMock()
    space.name = space_name
    space.id = space_id

    resp = MagicMock()
    resp.spaces = [space]
    resp.pagination.next_cursor = None

    api = MagicMock()
    api.list_spaces.return_value = resp
    return api


def _make_spaces_api_ambiguous(space_name: str) -> MagicMock:
    """Return a SpacesApi mock where *space_name* matches two spaces."""
    s1 = MagicMock()
    s1.name = space_name
    s1.id = "space-id-1"

    s2 = MagicMock()
    s2.name = space_name
    s2.id = "space-id-2"

    resp = MagicMock()
    resp.spaces = [s1, s2]
    resp.pagination.next_cursor = None

    api = MagicMock()
    api.list_spaces.return_value = resp
    return api


def _make_projects_api(project_name: str, project_id: str) -> MagicMock:
    """Return a ProjectsApi mock that returns a single matching project."""
    project = MagicMock()
    project.name = project_name
    project.id = project_id

    resp = MagicMock()
    resp.projects = [project]
    resp.pagination.next_cursor = None

    api = MagicMock()
    api.list_projects.return_value = resp
    return api


def _make_datasets_api(dataset_name: str, dataset_id: str) -> MagicMock:
    """Return a DatasetsApi mock that returns a single matching dataset."""
    dataset = MagicMock()
    dataset.name = dataset_name
    dataset.id = dataset_id

    resp = MagicMock()
    resp.datasets = [dataset]
    resp.pagination.next_cursor = None

    api = MagicMock()
    api.list_datasets.return_value = resp
    return api


# ---------------------------------------------------------------------------
# TestFindProjectIdSpaceResolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindProjectIdSpaceResolution:
    """Tests that _find_project_id resolves a space name to an exact ID before
    the list call, preventing substring-match false positives.
    """

    def test_space_name_resolves_to_id_before_list(self) -> None:
        """When space is a name, _find_space_id is called and list_projects
        receives space_id=<resolved>, space_name=None — not space_name="team".
        """
        spaces_api = _make_spaces_api("team", _SPACE_ID)
        projects_api = _make_projects_api("my-project", _PROJECT_ID)

        result = _find_project_id(
            projects_api, spaces_api, "my-project", "team"
        )

        assert result == _PROJECT_ID
        spaces_api.list_spaces.assert_called_once()
        projects_api.list_projects.assert_called_once()
        call_kwargs = projects_api.list_projects.call_args.kwargs
        assert call_kwargs["space_id"] == _SPACE_ID
        assert call_kwargs["space_name"] is None

    def test_ambiguous_space_name_raises(self) -> None:
        """When _find_space_id raises AmbiguousNameError, _find_project_id
        propagates the error instead of silently picking the wrong space.
        """
        spaces_api = _make_spaces_api_ambiguous("team")
        projects_api = MagicMock()

        with pytest.raises(AmbiguousNameError):
            _find_project_id(projects_api, spaces_api, "my-project", "team")

        projects_api.list_projects.assert_not_called()

    def test_space_id_bypasses_space_lookup(self) -> None:
        """When space is a base64 ID, _find_space_id is NOT called."""
        spaces_api = MagicMock()
        projects_api = _make_projects_api("my-project", _PROJECT_ID)

        result = _find_project_id(
            projects_api, spaces_api, "my-project", _SPACE_ID
        )

        assert result == _PROJECT_ID
        spaces_api.list_spaces.assert_not_called()
        call_kwargs = projects_api.list_projects.call_args.kwargs
        assert call_kwargs["space_id"] == _SPACE_ID
        assert call_kwargs["space_name"] is None

    def test_project_id_bypasses_both_lookups(self) -> None:
        """When project is a base64 ID, neither spaces nor projects API is called."""
        spaces_api = MagicMock()
        projects_api = MagicMock()

        result = _find_project_id(projects_api, spaces_api, _PROJECT_ID, "team")

        assert result == _PROJECT_ID
        spaces_api.list_spaces.assert_not_called()
        projects_api.list_projects.assert_not_called()

    def test_space_name_not_found_raises(self) -> None:
        """When the space name cannot be resolved, NotFoundError propagates."""
        resp = MagicMock()
        resp.spaces = []
        resp.pagination.next_cursor = None
        spaces_api = MagicMock()
        spaces_api.list_spaces.return_value = resp

        with pytest.raises(NotFoundError, match="space"):
            _find_project_id(
                MagicMock(), spaces_api, "my-project", "unknown-space"
            )


# ---------------------------------------------------------------------------
# TestFindDatasetIdSpaceResolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindDatasetIdSpaceResolution:
    """Tests that _find_dataset_id applies the same space-name-to-ID fix."""

    def test_space_name_resolves_to_id_before_list(self) -> None:
        """When space is a name, list_datasets receives space_id=<resolved>,
        space_name=None.
        """
        spaces_api = _make_spaces_api("team", _SPACE_ID)
        datasets_api = _make_datasets_api("my-dataset", _DATASET_ID)

        result = _find_dataset_id(
            datasets_api, spaces_api, "my-dataset", "team"
        )

        assert result == _DATASET_ID
        spaces_api.list_spaces.assert_called_once()
        datasets_api.list_datasets.assert_called_once()
        call_kwargs = datasets_api.list_datasets.call_args.kwargs
        assert call_kwargs["space_id"] == _SPACE_ID
        assert call_kwargs["space_name"] is None

    def test_ambiguous_space_name_raises(self) -> None:
        """When the space name is ambiguous, AmbiguousNameError propagates."""
        spaces_api = _make_spaces_api_ambiguous("team")
        datasets_api = MagicMock()

        with pytest.raises(AmbiguousNameError):
            _find_dataset_id(datasets_api, spaces_api, "my-dataset", "team")

        datasets_api.list_datasets.assert_not_called()

    def test_space_id_bypasses_space_lookup(self) -> None:
        """When space is a base64 ID, the spaces API is not called."""
        spaces_api = MagicMock()
        datasets_api = _make_datasets_api("my-dataset", _DATASET_ID)

        result = _find_dataset_id(
            datasets_api, spaces_api, "my-dataset", _SPACE_ID
        )

        assert result == _DATASET_ID
        spaces_api.list_spaces.assert_not_called()
        call_kwargs = datasets_api.list_datasets.call_args.kwargs
        assert call_kwargs["space_id"] == _SPACE_ID
        assert call_kwargs["space_name"] is None

    def test_dataset_id_bypasses_both_lookups(self) -> None:
        """When dataset is a base64 ID, neither spaces nor datasets API is called."""
        spaces_api = MagicMock()
        datasets_api = MagicMock()

        result = _find_dataset_id(datasets_api, spaces_api, _DATASET_ID, "team")

        assert result == _DATASET_ID
        spaces_api.list_spaces.assert_not_called()
        datasets_api.list_datasets.assert_not_called()


# ---------------------------------------------------------------------------
# _find_webhook_id
# ---------------------------------------------------------------------------


def _make_organizations_api(org_name: str, org_id: str) -> MagicMock:
    """Return an OrganizationsApi mock that resolves *org_name* to *org_id*."""
    org = MagicMock()
    org.name = org_name
    org.id = org_id

    resp = MagicMock()
    resp.organizations = [org]
    resp.pagination.next_cursor = None

    api = MagicMock()
    api.list_organizations.return_value = resp
    return api


def _make_webhooks_api(pages: list[list[tuple[str, str]]]) -> MagicMock:
    """Return a WebhooksApi mock whose list_webhooks yields *pages* in order.

    Each page is a list of ``(name, id)`` tuples. Every page but the last
    carries a ``next_cursor``.
    """
    responses = []
    for i, page in enumerate(pages):
        resp = MagicMock()
        resp.webhooks = []
        for name, wid in page:
            w = MagicMock()
            w.name = name
            w.id = wid
            resp.webhooks.append(w)
        resp.pagination.next_cursor = (
            f"cursor-{i + 1}" if i < len(pages) - 1 else None
        )
        responses.append(resp)

    api = MagicMock()
    api.list_webhooks.side_effect = responses
    return api


@pytest.mark.unit
class TestFindWebhookId:
    """Tests for _find_webhook_id."""

    def test_returns_id_unchanged_without_lookup(self) -> None:
        """A base64 webhook ID short-circuits every API call."""
        webhooks_api = MagicMock()
        orgs_api = MagicMock()

        result = _find_webhook_id(webhooks_api, orgs_api, _WEBHOOK_ID, None)

        assert result == _WEBHOOK_ID
        webhooks_api.list_webhooks.assert_not_called()
        orgs_api.list_organizations.assert_not_called()

    def test_name_without_organization_raises(self) -> None:
        """A name needs an organization to scope the lookup."""
        webhooks_api = MagicMock()

        with pytest.raises(NotFoundError) as excinfo:
            _find_webhook_id(webhooks_api, MagicMock(), "deploy", None)

        assert "Provide 'organization'" in str(excinfo.value)
        webhooks_api.list_webhooks.assert_not_called()

    def test_resolves_name_with_organization_id(self) -> None:
        """An organization ID is used directly as the org_id filter."""
        webhooks_api = _make_webhooks_api([[("deploy", _WEBHOOK_ID)]])
        orgs_api = MagicMock()

        result = _find_webhook_id(webhooks_api, orgs_api, "deploy", _ORG_ID)

        assert result == _WEBHOOK_ID
        orgs_api.list_organizations.assert_not_called()
        webhooks_api.list_webhooks.assert_called_once_with(
            org_id=_ORG_ID, name="deploy", limit=100, cursor=None
        )

    def test_resolves_organization_name_first(self) -> None:
        """An organization name is resolved to an ID before listing."""
        webhooks_api = _make_webhooks_api([[("deploy", _WEBHOOK_ID)]])
        orgs_api = _make_organizations_api("my-org", _ORG_ID)

        result = _find_webhook_id(webhooks_api, orgs_api, "deploy", "my-org")

        assert result == _WEBHOOK_ID
        assert webhooks_api.list_webhooks.call_args.kwargs["org_id"] == _ORG_ID

    def test_exact_match_skips_substring_matches(self) -> None:
        """The substring filter may return near misses; only exact wins."""
        webhooks_api = _make_webhooks_api(
            [[("deploy-staging", "other"), ("deploy", _WEBHOOK_ID)]]
        )

        result = _find_webhook_id(webhooks_api, MagicMock(), "deploy", _ORG_ID)

        assert result == _WEBHOOK_ID

    def test_pages_until_match(self) -> None:
        """Resolution follows next_cursor across pages."""
        webhooks_api = _make_webhooks_api(
            [[("deploy-a", "a")], [("deploy", _WEBHOOK_ID)]]
        )

        result = _find_webhook_id(webhooks_api, MagicMock(), "deploy", _ORG_ID)

        assert result == _WEBHOOK_ID
        assert webhooks_api.list_webhooks.call_count == 2
        assert (
            webhooks_api.list_webhooks.call_args_list[1].kwargs["cursor"]
            == "cursor-1"
        )

    def test_not_found_lists_available_names(self) -> None:
        """A miss raises NotFoundError carrying the names seen."""
        webhooks_api = _make_webhooks_api(
            [[("deploy-a", "a")], [("deploy-b", "b")]]
        )

        with pytest.raises(NotFoundError) as excinfo:
            _find_webhook_id(webhooks_api, MagicMock(), "deploy", _ORG_ID)

        assert excinfo.value.resource_type == "webhook"
        assert excinfo.value.available_names == ["deploy-a", "deploy-b"]
