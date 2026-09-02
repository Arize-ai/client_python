"""Unit tests for the space-name-to-ID fix in arize.utils.resolve."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from arize.exceptions.spaces import AmbiguousNameError
from arize.utils.resolve import (
    NotFoundError,
    _find_dataset_id,
    _find_project_id,
)

# A valid base64 identifier (decodes to "Space:9050:1JkR")
_SPACE_ID = "U3BhY2U6OTA1MDoxSmtS"

# A valid base64 identifier (decodes to "Project:123")
_PROJECT_ID = "UHJvamVjdDoxMjM="

# A valid base64 identifier (decodes to "Dataset:123")
_DATASET_ID = "RGF0YXNldDoxMjM="


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
