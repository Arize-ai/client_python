"""Tests for arize.audit_logs.types public re-exports."""

from __future__ import annotations

import pytest

import arize.audit_logs.types as types_module
from arize.audit_logs.types import (
    AuditLog,
    AuditLogListResponse,
    AuditLogOperationType,
    AuditLogsList200Response,
)


@pytest.mark.unit
class TestAuditLogsTypes:
    """Tests for the audit_logs types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        assert "AuditLog" in types_module.__all__
        assert "AuditLogListResponse" in types_module.__all__
        assert "AuditLogOperationType" in types_module.__all__
        assert "AuditLogsList200Response" in types_module.__all__

    def test_audit_log_operation_type_is_enum(self) -> None:
        from enum import Enum

        assert issubclass(AuditLogOperationType, Enum)

    def test_audit_log_is_class(self) -> None:
        assert isinstance(AuditLog, type)

    def test_audit_log_list_response_is_class(self) -> None:
        assert isinstance(AuditLogListResponse, type)

    def test_audit_log_list_response_is_alias_of_200_response(self) -> None:
        """AuditLogListResponse must be the same type as AuditLogsList200Response."""
        assert AuditLogListResponse is AuditLogsList200Response
