"""Client implementation for retrieving audit logs from the Arize platform."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arize.constants.config import DEFAULT_LIST_LIMIT
from arize.pre_releases import ReleaseStage, prerelease_endpoint

if TYPE_CHECKING:
    from datetime import datetime

    from arize._generated.api_client.api_client import ApiClient
    from arize.audit_logs.types import (
        AuditLogOperationType,
        ListAuditLogsResponse,
    )
    from arize.config import SDKConfiguration


class AuditLogsClient:
    """Client for retrieving Arize audit log entries.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The audit logs client is a thin wrapper around the generated REST API client,
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

        self._api = gen.AuditLogsApi(generated_client)

    @prerelease_endpoint(key="audit_logs.list", stage=ReleaseStage.BETA)
    def list(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        operation_type: AuditLogOperationType | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> ListAuditLogsResponse:
        """List audit log entries for the account, ordered newest first.

        The caller must be an account admin and the account must have audit
        logging enabled. Returns a 403 if neither condition is met.

        Args:
            start_time: Optional inclusive lower bound on ``created_at``
                (ISO 8601 datetime). When omitted, the server defaults to 30
                days before ``end_time``.
            end_time: Optional inclusive upper bound on ``created_at``
                (ISO 8601 datetime). When omitted, the server defaults to the
                current time.
            user_id: Optional base64-encoded user ID to filter results to
                entries for a specific user. When omitted, no user filtering
                is applied.
            operation_type: Optional operation type filter. When omitted, all
                operation types are returned.
            limit: Maximum number of entries to return (1-100).
            cursor: Opaque pagination cursor from a previous response.

        Returns:
            A paginated audit log list response from the Arize REST API.

        Raises:
            ApiException: If the API request fails (for example, the caller
                is not an account admin, or audit logging is not enabled).
        """
        return self._api.list_audit_logs(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            operation_type=operation_type,
            limit=limit,
            cursor=cursor,
        )
