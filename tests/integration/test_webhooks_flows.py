"""Integration tests for WebhooksClient end-to-end flows against the real Arize API.

Webhooks are organization-scoped. This module creates a throwaway
organization, space, and prompt as one-time module fixtures and tears them
down after the module runs; each test creates and cleans up its own webhook.

Run with:
    ARIZE_API_KEY=<key> \
        pytest tests/integration/test_webhooks_flows.py -m integration -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from arize._generated.api_client.exceptions import ApiException
from arize.utils.resolve import is_resource_id

API_KEY = os.environ.get("ARIZE_API_KEY", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not API_KEY,
        reason="ARIZE_API_KEY must be set",
    ),
]


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def arize_client() -> Any:
    from arize.client import ArizeClient

    return ArizeClient(api_key=API_KEY)


@pytest.fixture(scope="module")
def webhooks_client(arize_client: Any) -> Any:
    return arize_client.webhooks


@pytest.fixture(scope="module")
def organization_id(arize_client: Any) -> Any:
    org = arize_client.organizations.create(
        name=_unique("sdk-test-webhooks-org")
    )
    yield org.id
    arize_client.organizations.delete(organization=org.id)


@pytest.fixture(scope="module")
def prompt_id(arize_client: Any, organization_id: str) -> Any:
    from arize._generated import api_client as gen

    space = arize_client.spaces.create(
        name=_unique("sdk-test-webhooks-space"),
        organization_id=organization_id,
    )
    prompt = arize_client.prompts.create(
        space=space.id,
        name=_unique("sdk-test-webhooks-prompt"),
        commit_message="initial version",
        input_variable_format=gen.InputVariableFormat.F_STRING,
        provider=gen.LlmProvider.OPEN_AI,
        model="gpt-4o-mini",
        messages=[gen.LLMMessageRequest(role="USER", content="Hello {name}")],
    )
    yield prompt.id
    arize_client.prompts.delete(prompt=prompt.id)
    arize_client.spaces.delete(space=space.id)


class TestWebhooksCRUD:
    """End-to-end CRUD flows for WebhooksClient."""

    def test_create_get_delete_by_id(
        self, webhooks_client: Any, organization_id: str
    ) -> None:
        """Create a webhook, retrieve it by ID, then delete it."""
        name = _unique("sdk-test-webhook")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://example.com/hook",
        )
        try:
            assert webhook.name == name
            assert is_resource_id(webhook.id)

            fetched = webhooks_client.get(webhook=webhook.id)
            assert fetched.id == webhook.id
            assert fetched.name == name
        finally:
            webhooks_client.delete(webhook=webhook.id)

    def test_create_get_by_name(
        self, webhooks_client: Any, organization_id: str
    ) -> None:
        """Create a webhook, retrieve it by name."""
        name = _unique("sdk-test-webhook")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://example.com/hook",
        )
        try:
            fetched = webhooks_client.get(
                webhook=name, organization=organization_id
            )
            assert fetched.id == webhook.id
        finally:
            webhooks_client.delete(webhook=webhook.id)

    def test_create_hmac_returns_signing_secret_once(
        self, webhooks_client: Any, organization_id: str
    ) -> None:
        """HMAC_SHA256 create returns signing_secret; later reads only get the hint."""
        name = _unique("sdk-test-webhook-hmac")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://example.com/hook",
            auth_type="HMAC_SHA256",
        )
        try:
            assert webhook.signing_secret is not None
            assert webhook.signing_secret_hint is not None

            fetched = webhooks_client.get(webhook=webhook.id)
            assert fetched.signing_secret_hint is not None
        finally:
            webhooks_client.delete(webhook=webhook.id)

    def test_create_update(
        self, webhooks_client: Any, organization_id: str
    ) -> None:
        """Create a webhook then update its name and description."""
        name = _unique("sdk-test-webhook")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://example.com/hook",
        )
        updated_name = _unique("sdk-test-webhook-upd")
        try:
            updated = webhooks_client.update(
                webhook=webhook.id,
                name=updated_name,
                description="Updated by SDK integration test",
            )
            assert updated.name == updated_name
            assert updated.description == "Updated by SDK integration test"

            fetched = webhooks_client.get(webhook=webhook.id)
            assert fetched.name == updated_name
        finally:
            webhooks_client.delete(webhook=webhook.id)

    def test_create_appears_in_list(
        self, webhooks_client: Any, organization_id: str
    ) -> None:
        """Newly created webhook appears in list() results."""
        name = _unique("sdk-test-webhook")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://example.com/hook",
        )
        try:
            resp = webhooks_client.list(organization=organization_id, limit=100)
            assert webhook.id in [w.id for w in resp.webhooks]
        finally:
            webhooks_client.delete(webhook=webhook.id)

    def test_list_filter_by_name(
        self, webhooks_client: Any, organization_id: str
    ) -> None:
        """list(name=...) filters to webhooks whose names contain the substring."""
        name = _unique("sdk-test-webhook")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://example.com/hook",
        )
        try:
            resp = webhooks_client.list(organization=organization_id, name=name)
            assert any(w.id == webhook.id for w in resp.webhooks)
        finally:
            webhooks_client.delete(webhook=webhook.id)

    def test_create_missing_fields_raises_422(
        self, webhooks_client: Any, organization_id: str
    ) -> None:
        """Creating with empty required fields raises a typed 422."""
        with pytest.raises(ApiException) as exc_info:
            webhooks_client.create(
                organization=organization_id, name="", url=""
            )
        assert exc_info.value.status == 422


class TestWebhooksDelete:
    """End-to-end delete flow for WebhooksClient."""

    def test_create_delete_by_id(
        self, webhooks_client: Any, organization_id: str
    ) -> None:
        """Delete a webhook by ID; subsequent get raises a 404."""
        name = _unique("sdk-test-webhook")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://example.com/hook",
        )

        result = webhooks_client.delete(webhook=webhook.id)

        assert result is None
        with pytest.raises(ApiException) as exc_info:
            webhooks_client.get(webhook=webhook.id)
        assert exc_info.value.status == 404


class TestWebhooksTestAndDeliveryAttempts:
    """End-to-end flows for test() and list_delivery_attempts()."""

    def test_test_bearer_webhook_delivers(
        self, webhooks_client: Any, organization_id: str
    ) -> None:
        """test() on a BEARER webhook delivers to a real endpoint and reports its status."""
        name = _unique("sdk-test-webhook")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://httpbin.org/post",
        )
        try:
            result = webhooks_client.test(webhook=webhook.id)
            assert result.status_code == 200
        finally:
            webhooks_client.delete(webhook=webhook.id)

    def test_test_hmac_webhook_raises_400(
        self, webhooks_client: Any, organization_id: str
    ) -> None:
        """test() on an HMAC_SHA256 webhook raises a typed 400."""
        name = _unique("sdk-test-webhook-hmac")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://example.com/hook",
            auth_type="HMAC_SHA256",
        )
        try:
            with pytest.raises(ApiException) as exc_info:
                webhooks_client.test(webhook=webhook.id)
            assert exc_info.value.status == 400
        finally:
            webhooks_client.delete(webhook=webhook.id)

    def test_list_delivery_attempts_excludes_test_deliveries(
        self, webhooks_client: Any, organization_id: str
    ) -> None:
        """A test() delivery is a config smoke test and never shows up in list_delivery_attempts()."""
        name = _unique("sdk-test-webhook")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://httpbin.org/post",
        )
        try:
            webhooks_client.test(webhook=webhook.id)
            resp = webhooks_client.list_delivery_attempts(webhook=webhook.id)
            assert resp.delivery_attempts == []
        finally:
            webhooks_client.delete(webhook=webhook.id)


class TestWebhookSubscriptions:
    """End-to-end CRUD flows for webhook subscriptions on a prompt."""

    def test_create_get_list_delete_subscription(
        self, webhooks_client: Any, organization_id: str, prompt_id: str
    ) -> None:
        """Create a subscription, retrieve and list it, then delete it."""
        name = _unique("sdk-test-webhook")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://example.com/hook",
        )
        try:
            sub = webhooks_client.create_subscription(
                webhook=webhook.id,
                source_type="PROMPT",
                source_id=prompt_id,
                event="PROMPT_VERSION_CREATED",
            )
            try:
                assert is_resource_id(sub.id)

                fetched = webhooks_client.get_subscription(
                    subscription_id=sub.id
                )
                assert fetched.id == sub.id

                listed = webhooks_client.list_subscriptions(
                    source_type="PROMPT", source_id=prompt_id
                )
                assert any(s.id == sub.id for s in listed.subscriptions)
            finally:
                webhooks_client.delete_subscription(subscription_id=sub.id)
        finally:
            webhooks_client.delete(webhook=webhook.id)

    def test_duplicate_subscription_raises_409(
        self, webhooks_client: Any, organization_id: str, prompt_id: str
    ) -> None:
        """A duplicate (webhook, source, event) subscription raises a typed 409."""
        name = _unique("sdk-test-webhook")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://example.com/hook",
        )
        try:
            sub = webhooks_client.create_subscription(
                webhook=webhook.id,
                source_type="PROMPT",
                source_id=prompt_id,
                event="PROMPT_VERSION_CREATED",
            )
            try:
                with pytest.raises(ApiException) as exc_info:
                    webhooks_client.create_subscription(
                        webhook=webhook.id,
                        source_type="PROMPT",
                        source_id=prompt_id,
                        event="PROMPT_VERSION_CREATED",
                    )
                assert exc_info.value.status == 409
            finally:
                webhooks_client.delete_subscription(subscription_id=sub.id)
        finally:
            webhooks_client.delete(webhook=webhook.id)

    def test_delete_subscription_then_get_raises_404(
        self, webhooks_client: Any, organization_id: str, prompt_id: str
    ) -> None:
        """Deleting a subscription; a subsequent get raises a typed 404."""
        name = _unique("sdk-test-webhook")
        webhook = webhooks_client.create(
            organization=organization_id,
            name=name,
            url="https://example.com/hook",
        )
        try:
            sub = webhooks_client.create_subscription(
                webhook=webhook.id,
                source_type="PROMPT",
                source_id=prompt_id,
                event="PROMPT_VERSION_CREATED",
            )
            webhooks_client.delete_subscription(subscription_id=sub.id)
            with pytest.raises(ApiException) as exc_info:
                webhooks_client.get_subscription(subscription_id=sub.id)
            assert exc_info.value.status == 404
        finally:
            webhooks_client.delete(webhook=webhook.id)
