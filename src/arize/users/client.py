"""Client implementation for managing users in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.pre_releases import ReleaseStage, prerelease_endpoint

if TYPE_CHECKING:
    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration
    from arize.users.types import (
        CustomUserRole,
        InviteMode,
        PredefinedUserRole,
        User,
        UsersList200Response,
        UserStatus,
    )

logger = logging.getLogger(__name__)


class UsersClient:
    """Client for managing Arize users.

    Unlike organizations, users are looked up by ID only — not by name —
    because display names are not unique within an account.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The users client is a thin wrapper around the generated REST API
    client, using the shared generated API client owned by
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
        self._api = gen.UsersApi(generated_client)

    @prerelease_endpoint(key="users.list", stage=ReleaseStage.ALPHA)
    def list(
        self,
        *,
        email: str | None = None,
        status: list[UserStatus] | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> UsersList200Response:
        """List users in the account.

        This endpoint supports cursor-based pagination. When provided,
        ``email`` filters results to users whose email contains the
        given substring (case-insensitive).

        Args:
            email: Optional case-insensitive partial-match filter on email address.
            status: Optional list of statuses to filter by
                (e.g. ``["active", "invited"]``).
            limit: Maximum number of users to return (1-100). The server
                enforces an upper bound of 100.
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated user list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails.
        """
        return self._api.users_list(
            limit=limit,
            cursor=cursor,
            email=email,
            status=status,
        )

    @prerelease_endpoint(key="users.get", stage=ReleaseStage.ALPHA)
    def get(self, *, user_id: str) -> User:
        """Get a user by ID.

        Args:
            user_id: User ID to retrieve.

        Returns:
            The user object.

        Raises:
            ApiException: If the API request fails
                (for example, user not found).
        """
        return self._api.users_get(user_id=user_id)

    @prerelease_endpoint(key="users.create", stage=ReleaseStage.ALPHA)
    def create(
        self,
        *,
        name: str,
        email: str,
        role: PredefinedUserRole | CustomUserRole,
        invite_mode: InviteMode,
        is_developer: bool | None = None,
    ) -> User:
        """Create a new user.

        Args:
            name: Display name for the user (1-255 characters).
            email: Email address (used as the idempotency key).
            role: Account-level role assignment. Use
                ``PredefinedUserRole(name="<role>")`` for predefined roles
                (``admin``, ``member``, ``annotator``), or
                ``CustomUserRole(id="<role-id>")`` for custom RBAC roles.
            invite_mode: Invite mode (``"none"``, ``"email_link"``, or
                ``"temporary_password"``).
            is_developer: Whether the user should have developer permissions
                (can create GraphQL API keys). Defaults to ``True`` for
                ``admin`` and ``member`` roles, and ``False`` for
                ``annotator``.

        Returns:
            The created user object.

        Raises:
            ApiException: If the API request fails.
        """
        from arize._generated import api_client as gen

        kwargs = {}
        if is_developer is not None:
            kwargs["is_developer"] = is_developer

        body = gen.CreateUserRequest(
            name=name,
            email=email,
            role=gen.UserRoleAssignment(role._to_generated()),
            invite_mode=invite_mode,
            **kwargs,
        )
        return self._api.users_create(create_user_request=body)

    @prerelease_endpoint(key="users.update", stage=ReleaseStage.ALPHA)
    def update(
        self,
        *,
        user_id: str,
        name: str | None = None,
        is_developer: bool | None = None,
    ) -> User:
        """Update a user's metadata by ID.

        Args:
            user_id: User ID to update.
            name: Updated display name for the user.
            is_developer: Updated developer permission flag.

        Returns:
            The updated user object.

        Raises:
            ValueError: If neither ``name`` nor ``is_developer`` is provided.
            ApiException: If the API request fails
                (for example, user not found or insufficient permissions).
        """
        if name is None and is_developer is None:
            raise ValueError(
                "At least one of 'name' or 'is_developer' must be provided"
            )

        from arize._generated import api_client as gen

        body = gen.UserUpdate(
            name=name,
            is_developer=is_developer,
        )
        return self._api.users_update(user_id=user_id, user_update=body)

    @prerelease_endpoint(key="users.delete", stage=ReleaseStage.ALPHA)
    def delete(self, *, user_id: str) -> None:
        """Delete a user by ID.

        This operation soft-deletes the user and cascades to organization
        memberships, space memberships, API keys, and role bindings.

        Args:
            user_id: User ID to delete.

        Returns:
            This method returns None on success (204 No Content response).

        Raises:
            ApiException: If the API request fails
                (for example, user not found or insufficient
                permissions).
        """
        return self._api.users_delete(user_id=user_id)

    @prerelease_endpoint(
        key="users.resend_invitation", stage=ReleaseStage.ALPHA
    )
    def resend_invitation(self, *, user_id: str) -> None:
        """Resend an invitation email for a pending user.

        The target user must be in the ``invited`` state.

        Args:
            user_id: User ID to resend the invitation for.

        Returns:
            This method returns None on success (202 Accepted response).

        Raises:
            ApiException: If the API request fails
                (for example, user not found or user already active).
        """
        return self._api.users_resend_invitation(
            user_id=user_id,
            _headers={"Content-Type": "application/json"},
        )

    @prerelease_endpoint(key="users.reset_password", stage=ReleaseStage.ALPHA)
    def reset_password(self, *, user_id: str) -> None:
        """Trigger a password-reset email for a user.

        Generates a reset token and sends the user a password-reset email
        with a 30-minute link.

        The target user must authenticate via password (not SSO/SAML) and
        must have already verified their account.

        Args:
            user_id: User ID to send the password-reset email to.

        Returns:
            This method returns None on success (204 No Content response).

        Raises:
            ApiException: If the API request fails
                (for example, user not found, user authenticates via SSO,
                or user has not yet verified their account).
        """
        return self._api.users_password_reset(
            user_id=user_id,
            _headers={"Content-Type": "application/json"},
        )
