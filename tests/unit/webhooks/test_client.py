"""Unit tests for src/arize/webhooks/client.py."""

from __future__ import annotations

from unittest.mock import Mock, create_autospec, patch

import pytest

from arize._generated.api_client import WebhooksApi
from arize.utils.resolve import NotFoundError
from arize.webhooks.client import WebhooksClient
from arize.webhooks.types import (
    WebhookAuthType,
    WebhookEventType,
    WebhookSourceType,
)

# Base64 IDs that pass is_resource_id() — decode to "Type:123"
_WEBHOOK_ID = "V2ViaG9vazoxMjM="  # Webhook:123
_ORG_ID = "T3JnYW5pemF0aW9uOjEyMw=="  # Organization:123
_SUBSCRIPTION_ID = "V2ViaG9va1N1YnNjcmlwdGlvbjoxMjM="  # WebhookSubscription:123
_PROMPT_ID = "UHJvbXB0OjEyMw=="  # Prompt:123


@pytest.fixture
def mock_api() -> Mock:
    """Provide a mock WebhooksApi instance."""
    return create_autospec(WebhooksApi, instance=True)


@pytest.fixture
def mock_organizations_api() -> Mock:
    """Provide a mock OrganizationsApi instance."""
    return Mock()


@pytest.fixture
def webhooks_client(
    mock_sdk_config: Mock, mock_api: Mock, mock_organizations_api: Mock
) -> WebhooksClient:
    """Provide a WebhooksClient with mocked internals."""
    with (
        patch(
            "arize._generated.api_client.WebhooksApi",
            return_value=mock_api,
        ),
        patch(
            "arize._generated.api_client.OrganizationsApi",
            return_value=mock_organizations_api,
        ),
    ):
        return WebhooksClient(
            sdk_config=mock_sdk_config,
            generated_client=Mock(),
        )


def _org_list_response(name: str, org_id: str) -> Mock:
    """Return a list_organizations response resolving *name* to *org_id*."""
    org = Mock()
    org.name = name
    org.id = org_id
    resp = Mock()
    resp.organizations = [org]
    resp.pagination.next_cursor = None
    return resp


def _webhook_list_response(name: str, webhook_id: str) -> Mock:
    """Return a list_webhooks response resolving *name* to *webhook_id*."""
    webhook = Mock()
    webhook.name = name
    webhook.id = webhook_id
    resp = Mock()
    resp.webhooks = [webhook]
    resp.pagination.next_cursor = None
    return resp


@pytest.mark.unit
class TestWebhooksClientInit:
    """Tests for WebhooksClient initialisation."""

    def test_stores_sdk_config(
        self, mock_sdk_config: Mock, mock_api: Mock
    ) -> None:
        """Constructor must store sdk_config on the instance."""
        with (
            patch(
                "arize._generated.api_client.WebhooksApi",
                return_value=mock_api,
            ),
            patch("arize._generated.api_client.OrganizationsApi"),
        ):
            client = WebhooksClient(
                sdk_config=mock_sdk_config,
                generated_client=Mock(),
            )

        assert client._sdk_config is mock_sdk_config

    def test_creates_apis_with_generated_client(
        self, mock_sdk_config: Mock
    ) -> None:
        """Constructor must build both generated APIs from the shared client."""
        mock_generated_client = Mock()

        with (
            patch("arize._generated.api_client.WebhooksApi") as webhooks_cls,
            patch("arize._generated.api_client.OrganizationsApi") as orgs_cls,
        ):
            WebhooksClient(
                sdk_config=mock_sdk_config,
                generated_client=mock_generated_client,
            )

        webhooks_cls.assert_called_once_with(mock_generated_client)
        orgs_cls.assert_called_once_with(mock_generated_client)


@pytest.mark.unit
class TestWebhooksClientList:
    """Tests for WebhooksClient.list()."""

    def test_list_without_filters(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """list() with no arguments passes only the default limit."""
        webhooks_client.list()

        mock_api.list_webhooks.assert_called_once_with(
            org_id=None, name=None, limit=50, cursor=None
        )

    def test_list_with_organization_id(
        self,
        webhooks_client: WebhooksClient,
        mock_api: Mock,
        mock_organizations_api: Mock,
    ) -> None:
        """A base64 organization value is passed through without a lookup."""
        webhooks_client.list(
            organization=_ORG_ID, name="deploy", limit=10, cursor="c1"
        )

        mock_organizations_api.list_organizations.assert_not_called()
        mock_api.list_webhooks.assert_called_once_with(
            org_id=_ORG_ID, name="deploy", limit=10, cursor="c1"
        )

    def test_list_with_organization_name(
        self,
        webhooks_client: WebhooksClient,
        mock_api: Mock,
        mock_organizations_api: Mock,
    ) -> None:
        """An organization name is resolved to its ID before listing."""
        mock_organizations_api.list_organizations.return_value = (
            _org_list_response("my-org", _ORG_ID)
        )

        webhooks_client.list(organization="my-org")

        mock_api.list_webhooks.assert_called_once_with(
            org_id=_ORG_ID, name=None, limit=50, cursor=None
        )

    def test_list_returns_api_response(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """list() returns the generated API's response unchanged."""
        sentinel = Mock()
        mock_api.list_webhooks.return_value = sentinel

        assert webhooks_client.list() is sentinel


@pytest.mark.unit
class TestWebhooksClientGet:
    """Tests for WebhooksClient.get()."""

    def test_get_by_id(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """A base64 webhook ID skips resolution."""
        webhooks_client.get(webhook=_WEBHOOK_ID)

        mock_api.list_webhooks.assert_not_called()
        mock_api.get_webhook.assert_called_once_with(webhook_id=_WEBHOOK_ID)

    def test_get_by_name_with_organization(
        self,
        webhooks_client: WebhooksClient,
        mock_api: Mock,
        mock_organizations_api: Mock,
    ) -> None:
        """A webhook name is resolved through the org-scoped list endpoint."""
        mock_organizations_api.list_organizations.return_value = (
            _org_list_response("my-org", _ORG_ID)
        )
        mock_api.list_webhooks.return_value = _webhook_list_response(
            "deploy-notifier", _WEBHOOK_ID
        )

        webhooks_client.get(webhook="deploy-notifier", organization="my-org")

        mock_api.list_webhooks.assert_called_once_with(
            org_id=_ORG_ID, name="deploy-notifier", limit=100, cursor=None
        )
        mock_api.get_webhook.assert_called_once_with(webhook_id=_WEBHOOK_ID)

    def test_get_by_name_without_organization_raises(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """A webhook name without an organization cannot be resolved."""
        with pytest.raises(NotFoundError, match="Provide 'organization'"):
            webhooks_client.get(webhook="deploy-notifier")

        mock_api.get_webhook.assert_not_called()


@pytest.mark.unit
class TestWebhooksClientCreate:
    """Tests for WebhooksClient.create()."""

    def test_create_minimal(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """Required fields only; optional fields stay unset on the body."""
        webhooks_client.create(
            organization=_ORG_ID,
            name="deploy-notifier",
            url="https://example.com/hook",
        )

        body = mock_api.create_webhook.call_args.kwargs[
            "create_webhook_request"
        ]
        assert body.organization_id == _ORG_ID
        assert body.name == "deploy-notifier"
        assert body.url == "https://example.com/hook"
        assert body.description is None
        assert body.auth_type is None
        assert body.auth_token is None
        assert body.timeout_ms is None
        assert body.headers is None

    def test_create_with_all_fields(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """Every optional field is forwarded onto the request body."""
        webhooks_client.create(
            organization=_ORG_ID,
            name="deploy-notifier",
            url="https://example.com/hook",
            description="notify on deploy",
            auth_type=WebhookAuthType.BEARER,
            auth_token="Bearer tok",  # noqa: S106
            timeout_ms=15000,
            headers={"X-Team": "platform"},
        )

        body = mock_api.create_webhook.call_args.kwargs[
            "create_webhook_request"
        ]
        assert body.description == "notify on deploy"
        assert body.auth_type is WebhookAuthType.BEARER
        assert body.auth_token == "Bearer tok"  # noqa: S105
        assert body.timeout_ms == 15000
        assert body.headers == {"X-Team": "platform"}

    def test_create_resolves_organization_name(
        self,
        webhooks_client: WebhooksClient,
        mock_api: Mock,
        mock_organizations_api: Mock,
    ) -> None:
        """An organization name becomes organization_id on the body."""
        mock_organizations_api.list_organizations.return_value = (
            _org_list_response("my-org", _ORG_ID)
        )

        webhooks_client.create(
            organization="my-org",
            name="deploy-notifier",
            url="https://example.com/hook",
        )

        body = mock_api.create_webhook.call_args.kwargs[
            "create_webhook_request"
        ]
        assert body.organization_id == _ORG_ID

    def test_create_returns_api_response(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """create() returns the response carrying the one-time secret."""
        sentinel = Mock()
        mock_api.create_webhook.return_value = sentinel

        result = webhooks_client.create(
            organization=_ORG_ID, name="n", url="https://example.com"
        )

        assert result is sentinel


@pytest.mark.unit
class TestWebhooksClientUpdate:
    """Tests for WebhooksClient.update()."""

    def test_update_requires_a_field(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """update() with no fields raises before any API call."""
        with pytest.raises(ValueError, match="At least one of"):
            webhooks_client.update(webhook=_WEBHOOK_ID)

        mock_api.update_webhook.assert_not_called()

    def test_update_single_field(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """Only the provided field lands on the body."""
        webhooks_client.update(webhook=_WEBHOOK_ID, name="renamed")

        call = mock_api.update_webhook.call_args
        assert call.kwargs["webhook_id"] == _WEBHOOK_ID
        body = call.kwargs["update_webhook_request"]
        assert body.to_dict() == {"name": "renamed"}

    def test_update_all_fields(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """All supported fields are forwarded."""
        webhooks_client.update(
            webhook=_WEBHOOK_ID,
            name="renamed",
            description="new text",
            url="https://example.com/v2",
            auth_token="Bearer new",  # noqa: S106
            timeout_ms=20000,
            headers={"X-A": "1"},
        )

        body = mock_api.update_webhook.call_args.kwargs[
            "update_webhook_request"
        ]
        assert body.to_dict() == {
            "name": "renamed",
            "description": "new text",
            "url": "https://example.com/v2",
            "auth_token": "Bearer new",
            "timeout_ms": 20000,
            "headers": {"X-A": "1"},
        }

    def test_update_clears_description_with_none(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """An explicit None description serializes as JSON null."""
        webhooks_client.update(webhook=_WEBHOOK_ID, description=None)

        body = mock_api.update_webhook.call_args.kwargs[
            "update_webhook_request"
        ]
        assert body.to_dict() == {"description": None}

    def test_update_resolves_webhook_name(
        self,
        webhooks_client: WebhooksClient,
        mock_api: Mock,
        mock_organizations_api: Mock,
    ) -> None:
        """A webhook name is resolved before the update call."""
        mock_organizations_api.list_organizations.return_value = (
            _org_list_response("my-org", _ORG_ID)
        )
        mock_api.list_webhooks.return_value = _webhook_list_response(
            "deploy-notifier", _WEBHOOK_ID
        )

        webhooks_client.update(
            webhook="deploy-notifier", organization="my-org", url="https://x"
        )

        assert (
            mock_api.update_webhook.call_args.kwargs["webhook_id"]
            == _WEBHOOK_ID
        )


@pytest.mark.unit
class TestWebhooksClientDelete:
    """Tests for WebhooksClient.delete()."""

    def test_delete_by_id(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """delete() forwards the ID and returns the API's None."""
        mock_api.delete_webhook.return_value = None

        assert webhooks_client.delete(webhook=_WEBHOOK_ID) is None
        mock_api.delete_webhook.assert_called_once_with(webhook_id=_WEBHOOK_ID)

    def test_delete_by_name(
        self,
        webhooks_client: WebhooksClient,
        mock_api: Mock,
        mock_organizations_api: Mock,
    ) -> None:
        """delete() resolves a name through the organization."""
        mock_organizations_api.list_organizations.return_value = (
            _org_list_response("my-org", _ORG_ID)
        )
        mock_api.list_webhooks.return_value = _webhook_list_response(
            "deploy-notifier", _WEBHOOK_ID
        )

        webhooks_client.delete(webhook="deploy-notifier", organization="my-org")

        mock_api.delete_webhook.assert_called_once_with(webhook_id=_WEBHOOK_ID)


@pytest.mark.unit
class TestWebhooksClientTest:
    """Tests for WebhooksClient.test()."""

    def test_test_by_id(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """test() forwards the ID and returns the outcome."""
        sentinel = Mock()
        mock_api.test_webhook.return_value = sentinel

        assert webhooks_client.test(webhook=_WEBHOOK_ID) is sentinel
        mock_api.test_webhook.assert_called_once_with(webhook_id=_WEBHOOK_ID)


@pytest.mark.unit
class TestWebhooksClientListDeliveryAttempts:
    """Tests for WebhooksClient.list_delivery_attempts()."""

    def test_list_delivery_attempts_defaults(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """Default limit and no cursor are passed through."""
        webhooks_client.list_delivery_attempts(webhook=_WEBHOOK_ID)

        mock_api.list_webhook_delivery_attempts.assert_called_once_with(
            webhook_id=_WEBHOOK_ID, limit=50, cursor=None
        )

    def test_list_delivery_attempts_with_paging(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """Explicit limit and cursor are forwarded."""
        webhooks_client.list_delivery_attempts(
            webhook=_WEBHOOK_ID, limit=200, cursor="c2"
        )

        mock_api.list_webhook_delivery_attempts.assert_called_once_with(
            webhook_id=_WEBHOOK_ID, limit=200, cursor="c2"
        )


@pytest.mark.unit
class TestWebhooksClientListSubscriptions:
    """Tests for WebhooksClient.list_subscriptions()."""

    def test_list_subscriptions_unfiltered(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """No source filter lists across all readable sources."""
        webhooks_client.list_subscriptions()

        mock_api.list_webhook_subscriptions.assert_called_once_with(
            source_type=None, source_id=None, limit=50, cursor=None
        )

    def test_list_subscriptions_by_source(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """Paired source_type and source_id are forwarded."""
        webhooks_client.list_subscriptions(
            source_type=WebhookSourceType.PROMPT,
            source_id=_PROMPT_ID,
            limit=20,
            cursor="c3",
        )

        mock_api.list_webhook_subscriptions.assert_called_once_with(
            source_type=WebhookSourceType.PROMPT,
            source_id=_PROMPT_ID,
            limit=20,
            cursor="c3",
        )

    @pytest.mark.parametrize(
        ("source_type", "source_id"),
        [
            (WebhookSourceType.PROMPT, None),
            (None, _PROMPT_ID),
        ],
    )
    def test_list_subscriptions_rejects_half_filter(
        self,
        webhooks_client: WebhooksClient,
        mock_api: Mock,
        source_type: WebhookSourceType | None,
        source_id: str | None,
    ) -> None:
        """Giving only one half of the source filter raises locally."""
        with pytest.raises(ValueError, match="provided together"):
            webhooks_client.list_subscriptions(
                source_type=source_type, source_id=source_id
            )

        mock_api.list_webhook_subscriptions.assert_not_called()


@pytest.mark.unit
class TestWebhooksClientCreateSubscription:
    """Tests for WebhooksClient.create_subscription()."""

    def test_create_subscription_by_webhook_id(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """All fields land on the request body."""
        webhooks_client.create_subscription(
            webhook=_WEBHOOK_ID,
            source_type=WebhookSourceType.PROMPT,
            source_id=_PROMPT_ID,
            event=WebhookEventType.PROMPT_VERSION_CREATED,
        )

        body = mock_api.create_webhook_subscription.call_args.kwargs[
            "create_webhook_subscription_request"
        ]
        assert body.webhook_id == _WEBHOOK_ID
        assert body.source_type is WebhookSourceType.PROMPT
        assert body.source_id == _PROMPT_ID
        assert body.event is WebhookEventType.PROMPT_VERSION_CREATED

    def test_create_subscription_by_webhook_name(
        self,
        webhooks_client: WebhooksClient,
        mock_api: Mock,
        mock_organizations_api: Mock,
    ) -> None:
        """A webhook name is resolved to an ID for the body."""
        mock_organizations_api.list_organizations.return_value = (
            _org_list_response("my-org", _ORG_ID)
        )
        mock_api.list_webhooks.return_value = _webhook_list_response(
            "deploy-notifier", _WEBHOOK_ID
        )

        webhooks_client.create_subscription(
            webhook="deploy-notifier",
            organization="my-org",
            source_type=WebhookSourceType.EVALUATOR,
            source_id=_PROMPT_ID,
            event=WebhookEventType.EVALUATOR_VERSION_CREATED,
        )

        body = mock_api.create_webhook_subscription.call_args.kwargs[
            "create_webhook_subscription_request"
        ]
        assert body.webhook_id == _WEBHOOK_ID

    def test_create_subscription_returns_api_response(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """create_subscription() returns the API's response unchanged."""
        sentinel = Mock()
        mock_api.create_webhook_subscription.return_value = sentinel

        result = webhooks_client.create_subscription(
            webhook=_WEBHOOK_ID,
            source_type=WebhookSourceType.PROMPT,
            source_id=_PROMPT_ID,
            event=WebhookEventType.PROMPT_VERSION_LABELED,
        )

        assert result is sentinel


@pytest.mark.unit
class TestWebhooksClientSubscriptionById:
    """Tests for get_subscription() and delete_subscription()."""

    def test_get_subscription(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """get_subscription() forwards the ID."""
        sentinel = Mock()
        mock_api.get_webhook_subscription.return_value = sentinel

        result = webhooks_client.get_subscription(
            subscription_id=_SUBSCRIPTION_ID
        )

        assert result is sentinel
        mock_api.get_webhook_subscription.assert_called_once_with(
            subscription_id=_SUBSCRIPTION_ID
        )

    def test_delete_subscription(
        self, webhooks_client: WebhooksClient, mock_api: Mock
    ) -> None:
        """delete_subscription() forwards the ID and returns None."""
        mock_api.delete_webhook_subscription.return_value = None

        result = webhooks_client.delete_subscription(
            subscription_id=_SUBSCRIPTION_ID
        )

        assert result is None
        mock_api.delete_webhook_subscription.assert_called_once_with(
            subscription_id=_SUBSCRIPTION_ID
        )
