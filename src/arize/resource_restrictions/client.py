"""Client implementation for managing resource restrictions in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.pre_releases import ReleaseStage, prerelease_endpoint

if TYPE_CHECKING:
    from arize._generated.api_client import models
    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration

logger = logging.getLogger(__name__)


class ResourceRestrictionsClient:
    """Client for managing Arize resource restrictions.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    Resource restrictions prevent roles bound at higher hierarchy levels (space,
    org, account) from granting access to the restricted resource. Only space
    admins or users with the ``PROJECT_RESTRICT`` permission can restrict or
    unrestrict a resource.

    Currently only ``PROJECT`` resources are supported.

    The resource restrictions client is a thin wrapper around the generated REST
    API client, using the shared generated API client owned by
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
        self._api = gen.ResourceRestrictionsApi(generated_client)

    @prerelease_endpoint(
        key="resource_restrictions.restrict", stage=ReleaseStage.ALPHA
    )
    def restrict(self, *, resource_id: str) -> models.ResourceRestriction:
        """Mark a resource as restricted.

        Restricting a resource prevents roles bound at higher hierarchy levels
        (space, org, account) from granting access. Only space admins or users
        with the ``PROJECT_RESTRICT`` permission can perform this action.

        This operation is idempotent — restricting an already-restricted resource
        returns the existing restriction without error.

        Currently only ``PROJECT`` resources are supported.

        Args:
            resource_id: Global ID of the resource to restrict. Must encode a
                project resource ID.

        Returns:
            The resource restriction object.

        Raises:
            ApiException: If the API request fails (for example, unsupported
                resource type or insufficient permissions).
        """
        from arize._generated import api_client as gen

        body = gen.ResourceRestrictionsCreateRequest(resource_id=resource_id)
        return self._api.resource_restrictions_create(body).resource_restriction

    @prerelease_endpoint(
        key="resource_restrictions.unrestrict", stage=ReleaseStage.ALPHA
    )
    def unrestrict(self, *, resource_id: str) -> None:
        """Remove restriction from a resource.

        Removing a restriction means that roles bound at other levels of the
        hierarchy (space, org, account) can once again grant access to the
        resource.

        Args:
            resource_id: Global ID of the resource to unrestrict.

        Raises:
            ApiException: If the API request fails (for example, resource is
                not restricted, or insufficient permissions).
        """
        return self._api.resource_restrictions_delete(resource_id=resource_id)
