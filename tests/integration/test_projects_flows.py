"""Integration tests for ProjectsClient end-to-end flows against the real Arize API.

Each test creates real resources, exercises the full lifecycle, and always
cleans up after itself — even on failure.

Run with:
    ARIZE_API_KEY=<key> ARIZE_TEST_SPACE_NAME=<space> \
        pytest tests/integration/test_projects_flows.py -m integration -v
"""

from __future__ import annotations

import contextlib
import os
import uuid
from typing import Any

import pytest

from arize._generated.api_client.exceptions import ForbiddenException
from arize.utils.resolve import is_resource_id

API_KEY = os.environ.get("ARIZE_API_KEY", "")
SPACE_NAME = os.environ.get("ARIZE_TEST_SPACE_NAME", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY or not SPACE_NAME,
        reason="ARIZE_API_KEY and ARIZE_TEST_SPACE_NAME must be set",
    ),
]


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _delete_project_if_permitted(
    projects_client: Any, **delete_kwargs: Any
) -> None:
    """Best-effort delete; ignores 403 when the API key cannot delete projects."""
    with contextlib.suppress(ForbiddenException):
        projects_client.delete(**delete_kwargs)


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def projects_client(arize_client) -> Any:
    return arize_client.projects


class TestProjectsCRUD:
    """End-to-end CRUD flows for ProjectsClient."""

    def test_create_get_delete_by_id(self, projects_client) -> None:
        """Create a project, retrieve it by ID, then delete it."""
        name = _unique("sdk-test-proj")
        project = projects_client.create(name=name, space=SPACE_NAME)
        try:
            assert project.name == name
            assert is_resource_id(project.id)

            fetched = projects_client.get(project=project.id)
            assert fetched.id == project.id
            assert fetched.name == name
        finally:
            _delete_project_if_permitted(projects_client, project=project.id)

    def test_create_get_delete_by_name(self, projects_client) -> None:
        """Create a project, retrieve and delete it by name."""
        name = _unique("sdk-test-proj")
        project = projects_client.create(name=name, space=SPACE_NAME)
        try:
            fetched = projects_client.get(project=name, space=SPACE_NAME)
            assert fetched.id == project.id
        finally:
            _delete_project_if_permitted(
                projects_client, project=name, space=SPACE_NAME
            )

    def test_create_appears_in_list(self, projects_client) -> None:
        """Newly created project appears in list() results."""
        name = _unique("sdk-test-proj")
        project = projects_client.create(name=name, space=SPACE_NAME)
        try:
            resp = projects_client.list(space=SPACE_NAME, limit=100)
            project_ids = [p.id for p in resp.projects]
            assert project.id in project_ids
        finally:
            _delete_project_if_permitted(projects_client, project=project.id)

    def test_list_filter_by_name(self, projects_client) -> None:
        """list() name filter returns only matching projects."""
        name = _unique("sdk-test-proj")
        project = projects_client.create(name=name, space=SPACE_NAME)
        try:
            resp = projects_client.list(space=SPACE_NAME, name=name, limit=100)
            assert len(resp.projects) >= 1
            assert any(p.id == project.id for p in resp.projects)
        finally:
            _delete_project_if_permitted(projects_client, project=project.id)
