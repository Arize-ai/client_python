"""Client implementation for managing users in the Arize platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from arize.constants.config import DEFAULT_LIST_LIMIT
from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.users.types import (
    BulkUserDeletionResult,
    DeletionStatus,
    ListUsersResponse,
    User,
)
from arize.utils.resolve import NotFoundError, _find_user_id_by_email

if TYPE_CHECKING:
    import builtins

    from arize._generated.api_client.api_client import ApiClient
    from arize.config import SDKConfiguration
    from arize.users.types import (
        CustomUserRole,
        InviteMode,
        PredefinedUserRole,
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

    @prerelease_endpoint(key="users.list", stage=ReleaseStage.BETA)
    def list(
        self,
        *,
        email: str | None = None,
        status: list[UserStatus] | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> ListUsersResponse:
        """List users in the account.

        This endpoint supports cursor-based pagination. When provided,
        ``email`` filters results to users whose email contains the
        given substring (case-insensitive).

        Args:
            email: Optional case-insensitive partial-match filter on email address.
            status: Optional list of statuses to filter by
                (e.g. ``["ACTIVE", "INVITED"]``).
            limit: Maximum number of users to return (1-100). The server
                enforces an upper bound of 100.
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated user list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails.
        """
        return ListUsersResponse.model_validate(
            self._api.list_users(
                limit=limit,
                cursor=cursor,
                email=email,
                status=status,
            ),
            from_attributes=True,
        )

    @prerelease_endpoint(key="users.get", stage=ReleaseStage.BETA)
    def get(self, *, user: str) -> User | None:
        """Get a user by ID or email address.

        When ``user`` contains ``@`` it is treated as an email address:
        :meth:`list` is called with the email as a filter, and the first
        result whose address matches exactly (case-insensitive) is
        returned. ``None`` is returned when no such user exists.

        When ``user`` does not contain ``@`` it is treated as a user ID
        and the API is called directly. An :class:`ApiException` is raised
        if the user is not found.

        Args:
            user: User ID or exact email address to retrieve.

        Returns:
            The matching :class:`~arize.users.types.User`, or ``None``
            when ``user`` is an email address and no match exists.

        Raises:
            ApiException: If the API request fails
                (for example, user not found when querying by ID).
        """
        if "@" in user:
            response = self.list(email=user)
            needle = user.lower()
            for u in response.users:
                if u.email.lower() == needle:
                    return u
            return None
        return User.model_validate(
            self._api.get_user(user_id=user), from_attributes=True
        )

    @prerelease_endpoint(key="users.create", stage=ReleaseStage.BETA)
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
                (``ADMIN``, ``MEMBER``, ``ANNOTATOR``), or
                ``CustomUserRole(id="<role-id>")`` for custom RBAC roles.
            invite_mode: Invite mode (``"NONE"``, ``"EMAIL_LINK"``, or
                ``"TEMPORARY_PASSWORD"``).
            is_developer: Whether the user should have developer permissions
                (can create GraphQL API keys). Defaults to ``True`` for
                ``ADMIN`` and ``MEMBER`` roles, and ``False`` for
                ``ANNOTATOR``.

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
            role=gen.UserRoleAssignment(
                gen.PredefinedUserRoleAssignment(
                    type=gen.UserRoleAssignmentType.PREDEFINED,
                    name=role.name,
                )
                # String literal (not the enum) so mypy narrows the
                # discriminated union; drift is guarded by the test asserting
                # the Literal matches UserRoleAssignmentType.
                if role.type == "PREDEFINED"
                else gen.CustomUserRoleAssignment(
                    type=gen.UserRoleAssignmentType.CUSTOM,
                    id=role.id,
                )
            ),
            invite_mode=invite_mode,
            **kwargs,
        )
        return User.model_validate(
            self._api.create_user(create_user_request=body),
            from_attributes=True,
        )

    @prerelease_endpoint(key="users.update", stage=ReleaseStage.BETA)
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

        body = gen.UpdateUserRequest(
            name=name,
            is_developer=is_developer,
        )
        return User.model_validate(
            self._api.update_user(user_id=user_id, update_user_request=body),
            from_attributes=True,
        )

    @prerelease_endpoint(key="users.delete", stage=ReleaseStage.BETA)
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
        return self._api.delete_user(user_id=user_id)

    @prerelease_endpoint(key="users.resend_invitation", stage=ReleaseStage.BETA)
    def resend_invitation(self, *, user_id: str) -> None:
        """Resend an invitation email for a pending user.

        The target user must be in the ``INVITED`` state.

        Args:
            user_id: User ID to resend the invitation for.

        Returns:
            This method returns None on success (202 Accepted response).

        Raises:
            ApiException: If the API request fails
                (for example, user not found or user already active).
        """
        return self._api.resend_user_invitation(user_id=user_id)

    @prerelease_endpoint(key="users.bulk_delete", stage=ReleaseStage.BETA)
    def bulk_delete(
        self,
        *,
        user_ids: builtins.list[str] | None = None,
        emails: builtins.list[str] | None = None,
    ) -> builtins.list[BulkUserDeletionResult]:
        """Bulk-delete users by ID or email.

        At least one of ``user_ids`` or ``emails`` must be provided.
        When ``emails`` is given, each address is resolved to a user
        ID via the users list endpoint (case-insensitive exact match).

        Args:
            user_ids: User IDs to delete directly.
            emails: Email addresses to resolve and then delete.

        Returns:
            A list of :class:`~arize.users.types.BulkUserDeletionResult`
            recording the outcome of each deletion attempt.

        Raises:
            ValueError: If neither ``user_ids`` nor ``emails`` is
                provided.
        """
        if not user_ids and not emails:
            raise ValueError(
                "At least one of 'user_ids' or 'emails' must be provided"
            )

        results: builtins.list[BulkUserDeletionResult] = []
        ids_to_delete: builtins.list[str] = user_ids or []
        id_to_email: dict[str, str] = {}

        # Resolve emails to user IDs
        email: str
        for email in emails or []:
            try:
                uid = _find_user_id_by_email(self._api, email)
            except NotFoundError:
                logger.warning(
                    "No user found with email '%s' — skipping",
                    email,
                )
                results.append(
                    BulkUserDeletionResult(
                        user_id="",
                        email=email,
                        status=DeletionStatus.NOT_FOUND,
                        error=f"No user found with email '{email}'",
                    )
                )
                continue
            id_to_email[uid] = email
            ids_to_delete.append(uid)

        # Delete each user
        for uid in ids_to_delete:
            try:
                self.delete(user_id=uid)
                results.append(
                    BulkUserDeletionResult(
                        user_id=uid,
                        email=id_to_email.get(uid),
                        status=DeletionStatus.DELETED,
                    )
                )
            except Exception as exc:  # noqa: PERF203
                results.append(
                    BulkUserDeletionResult(
                        user_id=uid,
                        email=id_to_email.get(uid),
                        status=DeletionStatus.FAILED,
                        error=str(exc),
                    )
                )

        return results

    @prerelease_endpoint(key="users.reset_password", stage=ReleaseStage.BETA)
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
        return self._api.reset_user_password(user_id=user_id)
