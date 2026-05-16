"""Client implementation for managing API keys in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import _find_space_id

if TYPE_CHECKING:
    from datetime import datetime

    from arize._generated.api_client.api_client import ApiClient
    from arize.api_keys.types import (
        ApiKeyCreated,
        ApiKeysList200Response,
        ApiKeyStatus,
    )
    from arize.config import SDKConfiguration

SpaceRole = Literal["admin", "member", "read-only"]
OrgRole = Literal["admin", "member", "read-only"]
AccountRole = Literal["admin", "member"]

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
        self._spaces_api = gen.SpacesApi(generated_client)

    @prerelease_endpoint(key="api_keys.list", stage=ReleaseStage.ALPHA)
    def list(
        self,
        *,
        key_type: str | None = None,
        status: ApiKeyStatus | None = None,
        space: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ApiKeysList200Response:
        """List API keys.

        This endpoint supports cursor-based pagination. Optionally filter by
        ``key_type``, ``status``, ``space``, and ``user_id``.

        **Service keys** (``key_type="service"``): provide ``space`` to return
        all service keys for that space. When ``key_type`` is omitted alongside
        ``space``, service keys are returned implicitly. Optionally combine with
        ``user_id`` to filter by creator — available to any caller with space
        access.

        **User keys** (``key_type="user"``): returned by default (no ``space``).
        Provide ``user_id`` to view keys for a specific user — account admins
        only; non-admins receive a ``403``.

        Args:
            key_type: Optional key type filter (``"user"`` or ``"service"``).
            status: Optional status filter (``"active"`` or ``"deleted"``).
                Defaults to ``"active"`` on the server side when omitted.
            space: Space name or ID. When provided, filters to service keys for
                that space. Accepts a human-readable name or a base64 global ID.
            user_id: Base64 global ID of the user whose keys to return.
                For service keys (with ``space``), filters by creator and is
                available to any caller with space access. For user keys
                (without ``space``), requires account admin role.
            limit: Maximum number of keys to return (1 to 100). Defaults to 50.
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated API key list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails.
        """
        space_id = (
            _find_space_id(self._spaces_api, space)
            if space is not None
            else None
        )
        return self._api.api_keys_list(
            key_type=key_type,
            status=status,
            space_id=space_id,
            user_id=user_id,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="api_keys.create", stage=ReleaseStage.ALPHA)
    def create(
        self,
        *,
        name: str,
        description: str | None = None,
        expires_at: datetime | None = None,
    ) -> ApiKeyCreated:
        """Create a new user API key.

        Creates a user-type key that authenticates as the creating user with
        their full permissions. To create a space-scoped service key, use
        :meth:`create_service_key` instead.

        The returned ``ApiKeyCreated`` object contains the full raw key value
        in its ``key`` field. **This is the only time the raw key is
        returned.** Store it securely.

        Args:
            name: User-defined name for the API key (max 256 characters).
            description: Optional description (max 1000 characters).
            expires_at: Optional expiration timestamp. If omitted the key
                never expires. Must be a future timestamp.

        Returns:
            The created API key, including the one-time raw key value.

        Raises:
            ApiException: If the API request fails (e.g. invalid parameters
                or insufficient permissions).
        """
        from arize._generated import api_client as gen

        body = gen.ApiKeyCreate(
            name=name,
            description=description,
            key_type="user",
            expires_at=expires_at,
        )
        return self._api.api_keys_create(api_key_create=body)

    @prerelease_endpoint(
        key="api_keys.create_service_key", stage=ReleaseStage.ALPHA
    )
    def create_service_key(
        self,
        *,
        name: str,
        space: str,
        description: str | None = None,
        expires_at: datetime | None = None,
        space_role: SpaceRole | None = None,
        org_role: OrgRole | None = None,
        account_role: AccountRole | None = None,
    ) -> ApiKeyCreated:
        """Create a service-type API key for a space.

        Service keys are scoped to a specific space and backed by a dedicated
        bot user with configurable roles. When no roles are specified, the
        server applies its defaults (``space_role="member"``,
        ``org_role="read-only"``, ``account_role="member"``). All role
        assignments must be at or below the caller's own privilege level.

        The returned ``ApiKeyCreated`` object contains the full raw key value
        in its ``key`` field. **This is the only time the raw key is
        returned.** Store it securely.

        Args:
            name: User-defined name for the API key (max 256 characters).
            space: Space name or ID the service key is scoped to.
            description: Optional description (max 1000 characters).
            expires_at: Optional expiration timestamp. If omitted the key
                never expires. Must be a future timestamp.
            space_role: Role for the bot user within the space. One of
                ``"admin"``, ``"member"`` (default), or ``"read-only"``.
                Must be at or below the caller's own space role.
            org_role: Role for the bot user within the organization. One of
                ``"admin"``, ``"member"``, or ``"read-only"`` (default).
                Must be at or below the caller's own org role.
            account_role: Account-level role for the bot user. One of
                ``"admin"`` or ``"member"`` (default). Must be at or below
                the caller's own account role.

        Returns:
            The created API key, including the one-time raw key value.

        Raises:
            ApiException: If the API request fails (e.g. invalid role
                assignment or insufficient permissions).
        """
        from arize._generated import api_client as gen

        space_id = _find_space_id(self._spaces_api, space)

        roles = None
        if any(r is not None for r in (space_role, org_role, account_role)):
            roles = gen.ApiKeyRoles(
                space_role=space_role,
                org_role=org_role,
                account_role=account_role,
            )

        body = gen.ApiKeyCreate(
            name=name,
            description=description,
            key_type="service",
            expires_at=expires_at,
            space_id=space_id,
            roles=roles,
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
    ) -> ApiKeyCreated:
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
