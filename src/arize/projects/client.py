"""Client implementation for managing projects in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import (
    _find_project_id,
    _find_space_id,
    _resolve_resource,
)

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
        self._spaces_api = gen.SpacesApi(generated_client)

    @prerelease_endpoint(key="projects.list", stage=ReleaseStage.BETA)
    def list(
        self,
        *,
        name: str | None = None,
        space: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> models.ProjectsList200Response:
        """List projects the user has access to.

        This endpoint supports cursor-based pagination. When ``space`` is provided,
        results are filtered to that space.

        Args:
            name: Optional case-insensitive substring filter on the project name.
            space: Optional space filter. If the value is a base64-encoded resource ID it is
                treated as a space ID; otherwise it is used as a case-insensitive
                substring filter on the space name.
            limit: Maximum number of projects to return. The server may enforce
                an upper bound.
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated project list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails.
        """
        resolved_space = _resolve_resource(space)
        return self._api.projects_list(
            space_id=resolved_space.id,
            space_name=resolved_space.name,
            name=name,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="projects.create", stage=ReleaseStage.BETA)
    def create(
        self,
        *,
        name: str,
        space: str,
    ) -> models.Project:
        """Create a new project.

        Project names must be unique within the target space.

        Args:
            name: Project name (must be unique within the target space).
            space: Space ID or name to create the project in.

        Returns:
            The created project object.

        Raises:
            ApiException: If the API request fails
                (for example, due to invalid input or a uniqueness conflict).
        """
        space_id = _find_space_id(self._spaces_api, space)

        from arize._generated import api_client as gen

        body = gen.ProjectsCreateRequest(
            name=name,
            space_id=space_id,
        )
        return self._api.projects_create(projects_create_request=body)

    @prerelease_endpoint(key="projects.get", stage=ReleaseStage.BETA)
    def get(self, *, project: str, space: str | None = None) -> models.Project:
        """Get a project by ID or name.

        Args:
            project: Project ID or name.
            space: Space ID or name. Required when *project* is a name.

        Returns:
            The project object.

        Raises:
            ApiException: If the API request fails
                (for example, project not found).
        """
        project_id = _find_project_id(
            api=self._api,
            project=project,
            space=space,
        )
        return self._api.projects_get(project_id=project_id)

    @prerelease_endpoint(key="projects.delete", stage=ReleaseStage.BETA)
    def delete(self, *, project: str, space: str | None = None) -> None:
        """Delete a project by ID or name.

        This operation is irreversible.

        Args:
            project: Project ID or name.
            space: Space ID or name. Required when *project* is a name.

        Returns:
            This method returns None on success (common empty 204 response).

        Raises:
            ApiException: If the API request fails
                (for example, project not found or insufficient permissions).
        """
        project_id = _find_project_id(
            api=self._api,
            project=project,
            space=space,
        )
        return self._api.projects_delete(project_id=project_id)
