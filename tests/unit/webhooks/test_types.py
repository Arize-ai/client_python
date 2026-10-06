"""Tests for arize.webhooks.types public re-exports."""

from __future__ import annotations

from datetime import datetime, timezone
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


@pytest.mark.unit
class TestListResponsesToDf:
    def test_list_responses_have_to_df(self) -> None:
        for model in (
            ListWebhooksResponse,
            ListWebhookDeliveryAttemptsResponse,
            ListWebhookSubscriptionsResponse,
        ):
            assert callable(getattr(model, "to_df", None)), model.__name__

    def test_list_webhooks_to_df_one_row_per_webhook(self) -> None:
        response = ListWebhooksResponse(
            webhooks=[
                _webhook("wh_1", "Alpha"),
                _webhook("wh_2", "Beta"),
            ],
            pagination=PaginationMetadata(has_more=True, next_cursor="n"),
        )
        df = response.to_df()
        assert list(df["name"]) == ["Alpha", "Beta"]
        assert "pagination" not in df.columns

    def test_list_delivery_attempts_to_df_one_row_per_attempt(self) -> None:
        response = ListWebhookDeliveryAttemptsResponse(
            delivery_attempts=[
                WebhookDeliveryAttempt(
                    event_id="evt_1",
                    attempt_number=1,
                    payload={},
                    status_code=200,
                    created_at=_NOW,
                )
            ],
            pagination=PaginationMetadata(has_more=False),
        )
        df = response.to_df()
        assert list(df["event_id"]) == ["evt_1"]

    def test_list_subscriptions_to_df_one_row_per_subscription(self) -> None:
        response = ListWebhookSubscriptionsResponse(
            subscriptions=[
                WebhookSubscription(
                    id="sub_1",
                    webhook_id="wh_1",
                    source_type=WebhookSourceType.PROMPT,
                    source_id="pr_1",
                    event=WebhookEventType.PROMPT_VERSION_CREATED,
                    created_at=_NOW,
                )
            ],
            pagination=PaginationMetadata(has_more=False),
        )
        df = response.to_df()
        assert list(df["id"]) == ["sub_1"]


_NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _webhook(id: str, name: str) -> Webhook:
    return Webhook(
        id=id,
        organization_id="org_1",
        name=name,
        description="",
        url="https://example.com/hook",
        auth_type=WebhookAuthType.BEARER,
        timeout_ms=30000,
        created_at=_NOW,
        updated_at=_NOW,
    )
