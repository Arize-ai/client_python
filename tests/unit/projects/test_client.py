"""Unit tests for src/arize/projects/client.py."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from arize.projects.client import ProjectsClient


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock ProjectsApi instance."""
    return Mock()


@pytest.fixture
def projects_client(mock_sdk_config: Mock, mock_api: Mock) -> ProjectsClient:
    """Provide a ProjectsClient with mocked internals."""
    with patch(
        "arize._generated.api_client.ProjectsApi", return_value=mock_api
    ):
        return ProjectsClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestProjectsClientInit:
    """Tests for ProjectsClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.ProjectsApi", return_value=mock_api
        ):
            client = ProjectsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_projects_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to ProjectsApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.ProjectsApi"
        ) as mock_projects_api_cls:
            ProjectsClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_projects_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestProjectsClientList:
    """Tests for ProjectsClient.list()."""

    def test_list_with_space_id(
        self, projects_client: ProjectsClient, mock_api: Mock
    ) -> None:
        """list() should resolve a base64 resource ID space value to space_id."""
        projects_client.list(
            name="my-project",
            space="U3BhY2U6OTA1MDoxSmtS",
            limit=25,
            cursor="cursor-xyz",
        )

        mock_api.projects_list.assert_called_once_with(
            space_id="U3BhY2U6OTA1MDoxSmtS",
            space_name=None,
            name="my-project",
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_with_space_name(
        self, projects_client: ProjectsClient, mock_api: Mock
    ) -> None:
        """list() should resolve a non-prefixed space value to space_name."""
        projects_client.list(
            name="my-project",
            space="my-space",
            limit=25,
            cursor="cursor-xyz",
        )

        mock_api.projects_list.assert_called_once_with(
            space_id=None,
            space_name="my-space",
            name="my-project",
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_defaults(
        self, projects_client: ProjectsClient, mock_api: Mock
    ) -> None:
        """list() should default space/name/cursor to None and limit to 100."""
        projects_client.list()

        mock_api.projects_list.assert_called_once_with(
            space_id=None,
            space_name=None,
            name=None,
            limit=100,
            cursor=None,
        )

    def test_list_returns_api_response(
        self, projects_client: ProjectsClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from projects_list."""
        expected = Mock()
        mock_api.projects_list.return_value = expected

        result = projects_client.list()

        assert result is expected

    def test_list_emits_beta_prerelease_warning(
        self,
        projects_client: ProjectsClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """First call should emit the BETA prerelease warning."""
        from arize import pre_releases

        pre_releases._WARNED.clear()
        caplog.set_level(logging.WARNING)

        projects_client.list()

        assert any(
            "BETA" in record.message and "projects.list" in record.message
            for record in caplog.records
        )


@pytest.mark.unit
class TestProjectsClientCreate:
    """Tests for ProjectsClient.create()."""

    def test_create_resolves_space_and_builds_request(
        self, projects_client: ProjectsClient, mock_api: Mock
    ) -> None:
        """create() should resolve space, build ProjectCreate, and call projects_create."""
        with (
            patch(
                "arize.projects.client._find_space_id",
                return_value="resolved-space-id",
            ) as mock_resolve,
            patch(
                "arize._generated.api_client.ProjectCreate"
            ) as mock_request_cls,
        ):
            mock_request_cls.return_value = Mock()

            projects_client.create(name="my-project", space="my-space")

        mock_resolve.assert_called_once_with(
            projects_client._spaces_api, "my-space"
        )
        mock_request_cls.assert_called_once_with(
            name="my-project",
            space_id="resolved-space-id",
        )
        mock_api.projects_create.assert_called_once_with(
            project_create=mock_request_cls.return_value
        )

    def test_create_returns_api_response(
        self, projects_client: ProjectsClient, mock_api: Mock
    ) -> None:
        """create() should propagate the return value from projects_create."""
        expected = Mock()
        mock_api.projects_create.return_value = expected

        with (
            patch("arize.projects.client._find_space_id", return_value="sid"),
            patch("arize._generated.api_client.ProjectCreate"),
        ):
            result = projects_client.create(name="proj", space="space-id")

        assert result is expected
