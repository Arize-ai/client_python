"""Client implementation for managing API keys in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize._utils import unwrap_oneof
from arize.api_keys.types import ApiKeyType
from arize.constants.config import DEFAULT_LIST_LIMIT
from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import _find_space_id

if TYPE_CHECKING:
    import builtins
    from datetime import datetime

    from arize._generated.api_client.api_client import ApiClient
    from arize.api_keys.types import (
        ApiKeyStatus,
        ListApiKeysResponse,
        OrgBinding,
        RefreshApiKeyResponse,
        ServiceApiKeyCreated,
        UserApiKeyCreated,
        UserRoleAssignment,
    )
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
        self._spaces_api = gen.SpacesApi(generated_client)

    @prerelease_endpoint(key="api_keys.list", stage=ReleaseStage.BETA)
    def list(
        self,
        *,
        key_type: ApiKeyType | None = None,
        status: ApiKeyStatus | None = None,
        space: str | None = None,
        user_id: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> ListApiKeysResponse:
        """List API keys.

        This endpoint supports cursor-based pagination. Optionally filter by
        ``key_type``, ``status``, ``space``, and ``user_id``.

        **Service keys** (``key_type=ApiKeyType.SERVICE``): provide ``space`` to
        return all service keys for that space. When ``key_type`` is omitted
        alongside ``space``, service keys are returned implicitly. Optionally
        combine with ``user_id`` to filter by creator — available to any caller
        with space access.

        **User keys** (``key_type=ApiKeyType.USER``): returned by default (no
        ``space``). Provide ``user_id`` to view keys for a specific user —
        account admins only; non-admins receive a ``403``.

        Args:
            key_type: Optional key type filter (``ApiKeyType.USER`` or
                ``ApiKeyType.SERVICE``).
            status: Optional status filter (``ApiKeyStatus.ACTIVE`` or
                ``ApiKeyStatus.REVOKED``).
                Defaults to ``ApiKeyStatus.ACTIVE`` on the server side when omitted.
            space: Space name or ID. When provided, filters to service keys for
                that space. Accepts a human-readable name or a base64 identifier.
            user_id: Base64 identifier of the user whose keys to return.
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
        return self._api.list_api_keys(
            key_type=key_type,
            status=status,
            space_id=space_id,
            user_id=user_id,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="api_keys.create", stage=ReleaseStage.BETA)
    def create(
        self,
        *,
        name: str,
        description: str | None = None,
        expires_at: datetime | None = None,
    ) -> UserApiKeyCreated:
        """Create a new user API key.

        Creates a user-type key that authenticates as the creating user with
        their full permissions. To create a space-scoped service key, use
        :meth:`create_service_key` instead.

        The returned :class:`UserApiKeyCreated` object contains the full raw
        key value in its ``key`` field. **This is the only time the raw key
        is returned.** Store it securely.

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

        user_body = gen.CreateUserApiKeyRequest(
            key_type=ApiKeyType.USER,
            name=name,
            description=description,
            expires_at=expires_at,
        )
        body = gen.CreateApiKeyRequest(user_body)
        return unwrap_oneof(
            self._api.create_api_key(create_api_key_request=body)
        )  # type: ignore[return-value]

    @prerelease_endpoint(
        key="api_keys.create_service_key", stage=ReleaseStage.BETA
    )
    def create_service_key(
        self,
        *,
        name: str,
        orgs: builtins.list[OrgBinding],
        account_role: UserRoleAssignment | None = None,
        description: str | None = None,
        expires_at: datetime | None = None,
    ) -> ServiceApiKeyCreated:
        """Create a service-type API key with org and space bindings.

        Service keys are tied to a dedicated service account scoped to one or
        more organizations, each containing one or more spaces. All spaces must
        belong to the same account. When no role is specified for a space or
        org, the server applies the default predefined role (``MEMBER`` for
        spaces, ``READ_ONLY`` for orgs, ``MEMBER`` for accounts).

        The returned :class:`ServiceApiKeyCreated` object contains the full
        raw key value in its ``key`` field. **This is the only time the raw
        key is returned.** Store it securely.

        Args:
            name: User-defined name for the API key (max 256 characters).
            orgs: List of :class:`OrgBinding` objects, each specifying an
                optional org-level role and a list of :class:`SpaceBinding`
                objects within that org. At least one org with at least one
                space is required.
            account_role: Optional account-level role for the bot user. When
                ``None``, the server applies the default predefined ``MEMBER``
                role.
            description: Optional description (max 1000 characters).
            expires_at: Optional expiration timestamp. If omitted the key
                never expires. Must be a future timestamp.

        Returns:
            The created API key, including the one-time raw key value.

        Raises:
            ValueError: If ``orgs`` is empty or any org has no space bindings.
            ApiException: If the API request fails (e.g. invalid role
                assignment or insufficient permissions).
        """
        if not orgs:
            raise ValueError(
                "orgs must contain at least one entry for service keys"
            )

        from arize._generated import api_client as gen

        # Pre-validate and resolve all space IDs before writing any bindings.
        resolved_space_ids: dict[str, str] = {}
        for org_binding in orgs:
            for space_binding in org_binding.spaces:
                resolved_space_ids[space_binding.space] = _find_space_id(
                    self._spaces_api, space_binding.space
                )

        org_bindings_gen = []
        for org_binding in orgs:
            space_bindings_gen = []
            for binding in org_binding.spaces:
                space_id = resolved_space_ids[binding.space]
                space_bindings_gen.append(
                    gen.ServiceKeySpaceAssignment(
                        space_id=space_id,
                        role=binding.role,
                    )
                )
            org_bindings_gen.append(
                gen.ServiceKeyOrgAssignment(
                    org_id=org_binding.org_id,
                    role=org_binding.role,
                    spaces=space_bindings_gen,
                )
            )

        service_body = gen.CreateServiceApiKeyRequest(
            key_type=ApiKeyType.SERVICE,
            name=name,
            description=description,
            expires_at=expires_at,
            account_role=account_role,
            organizations=org_bindings_gen,
        )
        body = gen.CreateApiKeyRequest(service_body)
        return unwrap_oneof(
            self._api.create_api_key(create_api_key_request=body)
        )  # type: ignore[return-value]

    @prerelease_endpoint(key="api_keys.revoke", stage=ReleaseStage.BETA)
    def revoke(self, *, api_key_id: str) -> None:
        """Revoke an API key.

        The key's status is set to ``ApiKeyStatus.REVOKED`` and it is deactivated
        immediately and permanently. This operation is irreversible — the key
        will stop working right away. Revoking an already-revoked key is a
        no-op and still succeeds.

        Args:
            api_key_id: ID of the API key to revoke.

        Returns:
            None

        Raises:
            ApiException: If the API
                request fails (e.g. key not found or insufficient permissions).
        """
        return self._api.revoke_api_key(api_key_id=api_key_id)

    @prerelease_endpoint(key="api_keys.refresh", stage=ReleaseStage.BETA)
    def refresh(
        self,
        *,
        api_key_id: str,
        expires_at: datetime | None = None,
        grace_period_seconds: int | None = None,
    ) -> RefreshApiKeyResponse:
        """Refresh an existing API key.

        Atomically revokes the old key and issues a replacement with the
        same name, description, type, and scope.

        Use ``grace_period_seconds`` to keep the old key valid briefly while your
        services rotate to the new key.

        Args:
            api_key_id: ID of the API key to refresh.
            expires_at: New expiration for the replacement key. If omitted the
                replacement key will not expire.
            grace_period_seconds: Optional grace window, in seconds, during
                which the old key remains valid after refresh. If omitted or
                ``0``, the old key is invalidated immediately.

        Returns:
            The newly issued API key, including the one-time raw key value.

        Raises:
            ApiException: If the API
                request fails (e.g. key not found, already deleted, or
                insufficient permissions).
        """
        from arize._generated import api_client as gen

        body = gen.RefreshApiKeyRequest(
            expires_at=expires_at,
            grace_period_seconds=grace_period_seconds,
        )
        return self._api.refresh_api_key(
            api_key_id=api_key_id,
            refresh_api_key_request=body,
        )
