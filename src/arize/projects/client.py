"""Client implementation for managing projects in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.pre_releases import ReleaseStage, prerelease_endpoint

if TYPE_CHECKING:
    from arize._generated.api_client import models
    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration

logger = logging.getLogger(__name__)


class ProjectsClient:
    """Client for managing Arize projects and project-level operations.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The projects client is a thin wrapper around the generated REST API client,
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

        # Use the provided client directly
        self._api = gen.ProjectsApi(generated_client)

    @prerelease_endpoint(key="projects.list", stage=ReleaseStage.BETA)
    def list(
        self,
        *,
        space_id: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> models.ProjectsList200Response:
        """List projects the user has access to.

        This endpoint supports cursor-based pagination. When provided, `space_id`
        filters results to a particular space.

        Args:
            space_id: Optional space ID to filter results.
            limit: Maximum number of projects to return. The server may enforce
                an upper bound.
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated project list response from the Arize REST API.

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the API request fails.
        """
        return self._api.projects_list(
            space_id=space_id,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="projects.create", stage=ReleaseStage.BETA)
    def create(
        self,
        *,
        name: str,
        space_id: str,
    ) -> models.Project:
        """Create a new project.

        Project names must be unique within the target space.

        Args:
            name: Project name (must be unique within `space_id`).
            space_id: Space ID to create the project in.

        Returns:
            The created project object.

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the API request fails
                (for example, due to invalid input or a uniqueness conflict).
        """
        from arize._generated import api_client as gen

        body = gen.ProjectsCreateRequest(
            name=name,
            space_id=space_id,
        )
        return self._api.projects_create(projects_create_request=body)

    @prerelease_endpoint(key="projects.get", stage=ReleaseStage.BETA)
    def get(self, *, project_id: str) -> models.Project:
        """Get a project by ID.

        Args:
            project_id: Project ID.

        Returns:
            The project object.

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the API request fails
                (for example, project not found).
        """
        return self._api.projects_get(project_id=project_id)

    @prerelease_endpoint(key="projects.delete", stage=ReleaseStage.BETA)
    def delete(self, *, project_id: str) -> None:
        """Delete a project by ID.

        This operation is irreversible.

        Args:
            project_id: Project ID.

        Returns:
            This method returns None on success (common empty 204 response).

        Raises:
            arize._generated.api_client.exceptions.ApiException: If the API request fails
                (for example, project not found or insufficient permissions).
        """
        return self._api.projects_delete(project_id=project_id)
