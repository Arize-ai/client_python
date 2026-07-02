"""Unit tests for src/arize/audit_logs/client.py."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from arize.audit_logs.client import AuditLogsClient
from arize.audit_logs.types import AuditLogOperationType


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock AuditLogsApi instance."""
    return Mock()


@pytest.fixture
def audit_logs_client(mock_sdk_config: Mock, mock_api: Mock) -> AuditLogsClient:
    """Provide an AuditLogsClient with mocked internals."""
    with patch(
        "arize._generated.api_client.AuditLogsApi", return_value=mock_api
    ):
        return AuditLogsClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


@pytest.mark.unit
class TestAuditLogsClientInit:
    """Tests for AuditLogsClient.__init__()."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor should store sdk_config on the instance."""
        with patch(
            "arize._generated.api_client.AuditLogsApi", return_value=mock_api
        ):
            client = AuditLogsClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )
        assert client._sdk_config is mock_sdk_config

    def test_creates_audit_logs_api_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor should pass generated_client to AuditLogsApi."""
        mock_generated_client = Mock()
        with patch(
            "arize._generated.api_client.AuditLogsApi"
        ) as mock_audit_logs_api_cls:
            AuditLogsClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )
        mock_audit_logs_api_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestAuditLogsClientList:
    """Tests for AuditLogsClient.list()."""

    def test_list_defaults(
        self, audit_logs_client: AuditLogsClient, mock_api: Mock
    ) -> None:
        """list() should default all filter params to None and limit to 50."""
        audit_logs_client.list()

        mock_api.audit_logs_list.assert_called_once_with(
            start_time=None,
            end_time=None,
            user_id=None,
            operation_type=None,
            limit=50,
            cursor=None,
        )

    def test_list_with_all_filters(
        self, audit_logs_client: AuditLogsClient, mock_api: Mock
    ) -> None:
        """list() should forward all filter params to the generated API."""
        start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        end = datetime(2026, 5, 31, tzinfo=timezone.utc)

        audit_logs_client.list(
            start_time=start,
            end_time=end,
            user_id="VXNlcjoxMjM0NQ==",
            operation_type=AuditLogOperationType.MUTATION,
            limit=25,
            cursor="cursor-xyz",
        )

        mock_api.audit_logs_list.assert_called_once_with(
            start_time=start,
            end_time=end,
            user_id="VXNlcjoxMjM0NQ==",
            operation_type=AuditLogOperationType.MUTATION,
            limit=25,
            cursor="cursor-xyz",
        )

    def test_list_returns_api_response(
        self, audit_logs_client: AuditLogsClient, mock_api: Mock
    ) -> None:
        """list() should propagate the return value from audit_logs_list."""
        expected = Mock()
        mock_api.audit_logs_list.return_value = expected

        result = audit_logs_client.list()

        assert result is expected

    def test_list_with_operation_type_query(
        self, audit_logs_client: AuditLogsClient, mock_api: Mock
    ) -> None:
        """list() with QUERY operation_type should forward it correctly."""
        audit_logs_client.list(operation_type=AuditLogOperationType.QUERY)

        mock_api.audit_logs_list.assert_called_once_with(
            start_time=None,
            end_time=None,
            user_id=None,
            operation_type=AuditLogOperationType.QUERY,
            limit=50,
            cursor=None,
        )
