"""Tests for arize.webhooks.types public re-exports."""

from __future__ import annotations

from enum import Enum

import pytest

import arize.webhooks.types as types_module
from arize.webhooks.types import (
    CreateWebhookResponse,
    ListWebhookDeliveryAttemptsResponse,
    ListWebhooksResponse,
    ListWebhookSubscriptionsResponse,
    PaginationMetadata,
    TestWebhookResponse,
    Webhook,
    WebhookAuthType,
    WebhookDeliveryAttempt,
    WebhookEventType,
    WebhookSourceType,
    WebhookSubscription,
)


@pytest.mark.unit
class TestWebhooksTypes:
    """Tests for the webhooks types module re-exports."""

    def test_all_exports_are_accessible(self) -> None:
        """Every name in __all__ should be accessible as a module attribute."""
        for name in types_module.__all__:
            assert hasattr(types_module, name), f"{name} missing from module"
            assert getattr(types_module, name) is not None, f"{name} is None"

    def test_all_is_sorted(self) -> None:
        """__all__ is kept alphabetized."""
        assert list(types_module.__all__) == sorted(types_module.__all__)

    def test_expected_names_in_all(self) -> None:
        """__all__ should contain the expected public type names."""
        expected = {
            "CreateWebhookResponse",
            "ListWebhookDeliveryAttemptsResponse",
            "ListWebhookSubscriptionsResponse",
            "ListWebhooksResponse",
            "PaginationMetadata",
            "TestWebhookResponse",
            "Webhook",
            "WebhookAuthType",
            "WebhookDeliveryAttempt",
            "WebhookEventType",
            "WebhookSourceType",
            "WebhookSubscription",
        }
        assert expected == set(types_module.__all__)

    @pytest.mark.parametrize(
        "enum_cls", [WebhookAuthType, WebhookEventType, WebhookSourceType]
    )
    def test_enums_are_enums(self, enum_cls: type) -> None:
        assert issubclass(enum_cls, Enum)

    def test_event_type_values(self) -> None:
        """The event enum carries the four subscribable events."""
        assert {e.value for e in WebhookEventType} == {
            "PROMPT_VERSION_CREATED",
            "PROMPT_VERSION_LABELED",
            "PROMPT_VERSION_UNLABELED",
            "EVALUATOR_VERSION_CREATED",
        }

    @pytest.mark.parametrize(
        "cls",
        [
            CreateWebhookResponse,
            ListWebhookDeliveryAttemptsResponse,
            ListWebhooksResponse,
            ListWebhookSubscriptionsResponse,
            PaginationMetadata,
            TestWebhookResponse,
            Webhook,
            WebhookDeliveryAttempt,
            WebhookSubscription,
        ],
    )
    def test_type_is_class(self, cls: type) -> None:
        assert isinstance(cls, type)
