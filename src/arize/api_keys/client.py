"""Client implementation for managing API keys in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from arize.pre_releases import ReleaseStage, prerelease_endpoint

if TYPE_CHECKING:
    from datetime import datetime

    from arize._generated.api_client import models
    from arize._generated.api_client.api_client import ApiClient
    from arize._generated.api_client.models.api_key_status import ApiKeyStatus
    from arize.config import SDKConfiguration

logger = logging.getLogger(__name__)


class ApiKeysClient:
    """Client for managing Arize API keys.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The API keys client is a thin wrapper around the generated REST API client,
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

        from arize._generated import api_client as gen

        self._api = gen.APIKeysApi(generated_client)

    @prerelease_endpoint(key="api_keys.list", stage=ReleaseStage.ALPHA)
    def list(
        self,
        *,
        key_type: str | None = None,
        status: ApiKeyStatus | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> models.ApiKeysList200Response:
        """List API keys for the authenticated user.

        This endpoint supports cursor-based pagination. Optionally filter by
        ``key_type`` (``"user"`` or ``"service"``) and ``status``
        (``"active"`` or ``"deleted"``).

        Args:
            key_type: Optional key type filter (``"user"`` or ``"service"``).
            status: Optional status filter (``"active"`` or ``"deleted"``).
                Defaults to ``"active"`` on the server side when omitted.
            limit: Maximum number of keys to return (1 to 100). Defaults to 50.
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated API key list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails.
        """
        return self._api.api_keys_list(
            key_type=key_type,
            status=status,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="api_keys.create", stage=ReleaseStage.ALPHA)
    def create(
        self,
        *,
        name: str,
        description: str | None = None,
        key_type: Literal["user", "service"] = "user",
        expires_at: datetime | None = None,
        space_id: str | None = None,
    ) -> models.ApiKeyCreated:
        """Create a new API key.

        Two key types are supported:

        - ``"user"``: authenticates as the creating user with their full
          permissions. ``space_id`` must not be set.
        - ``"service"``: scoped to a specific space, backed by a dedicated
          bot user with limited roles. ``space_id`` is required.

        The returned :class:`~arize._generated.api_client.models.ApiKeyCreated`
        object contains the full raw key value in its ``key`` field. **This is
        the only time the raw key is returned.** Store it securely.

        Args:
            name: User-defined name for the API key (max 256 characters).
            description: Optional description (max 1000 characters).
            key_type: Type of key to create — ``"user"`` (default) or
                ``"service"``.
            expires_at: Optional expiration timestamp. If omitted the key
                never expires. Must be a future timestamp.
            space_id: Space ID the service key is scoped to. Required when
                ``key_type`` is ``"service"``; must not be set for user keys.

        Returns:
            The created API key, including the one-time raw key value.

        Raises:
            ApiException: If the API
                request fails (e.g. invalid parameters or insufficient
                permissions).
        """
        from arize._generated import api_client as gen

        body = gen.ApiKeyCreate(
            name=name,
            description=description,
            key_type=key_type,
            expires_at=expires_at,
            space_id=space_id,
        )
        return self._api.api_keys_create(api_key_create=body)

    @prerelease_endpoint(key="api_keys.delete", stage=ReleaseStage.ALPHA)
    def delete(self, *, api_key_id: str) -> None:
        """Delete an API key.

        The key is deactivated immediately and permanently. This operation is
        irreversible — the key will stop working right away.

        Args:
            api_key_id: ID of the API key to delete.

        Returns:
            None

        Raises:
            ApiException: If the API
                request fails (e.g. key not found or insufficient permissions).
        """
        return self._api.api_keys_delete(api_key_id=api_key_id)

    @prerelease_endpoint(key="api_keys.refresh", stage=ReleaseStage.ALPHA)
    def refresh(
        self,
        *,
        api_key_id: str,
        expires_at: datetime | None = None,
    ) -> models.ApiKeyCreated:
        """Refresh an existing API key.

        Atomically revokes the old key and issues a replacement with the same
        name, description, type, and scope. The new raw key value is returned
        in the ``key`` field of the response. **This is the only time the new
        raw key is returned.** Store it securely.

        Args:
            api_key_id: ID of the API key to refresh.
            expires_at: New expiration for the replacement key. If omitted the
                replacement key will not expire.

        Returns:
            The newly issued API key, including the one-time raw key value.

        Raises:
            ApiException: If the API
                request fails (e.g. key not found, already deleted, or
                insufficient permissions).
        """
        from arize._generated import api_client as gen

        body = gen.ApiKeyRefresh(expires_at=expires_at)
        return self._api.api_keys_refresh(
            api_key_id=api_key_id,
            api_key_refresh=body,
        )
