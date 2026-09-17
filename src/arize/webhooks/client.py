"""Client implementation for managing webhooks in the Arize platform."""

from __future__ import annotations

import logging
from typing import Any

from arize._generated.api_client.api_client import ApiClient  # noqa: TC001
from arize.config import SDKConfiguration  # noqa: TC001
from arize.constants.config import DEFAULT_LIST_LIMIT
from arize.pre_releases import ReleaseStage, prerelease_endpoint
from arize.utils.resolve import _find_organization_id, _find_webhook_id
from arize.utils.unset import _UNSET, UNSET, is_provided
from arize.webhooks.types import (  # noqa: TC001
    CreateWebhookResponse,
    ListWebhookDeliveryAttemptsResponse,
    ListWebhooksResponse,
    ListWebhookSubscriptionsResponse,
    TestWebhookResponse,
    Webhook,
    WebhookAuthType,
    WebhookEventType,
    WebhookSourceType,
    WebhookSubscription,
)

logger = logging.getLogger(__name__)


class WebhooksClient:
    """Client for managing webhooks and their subscriptions.

    A webhook is an organization-level destination: an HTTPS endpoint plus the
    authentication used to call it. A subscription delivers one event from one
    prompt or evaluator to one webhook.

    This class is primarily intended for internal use within the SDK. Users are
    highly encouraged to access resource-specific functionality via
    :class:`arize.ArizeClient`.

    The webhooks client is a thin wrapper around the generated REST API client,
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

        # Import at runtime so it's still lazy and extras-gated by the parent
        from arize._generated import api_client as gen

        self._api = gen.WebhooksApi(generated_client)
        self._organizations_api = gen.OrganizationsApi(generated_client)

    # ------------------------------------------------------------------
    # Webhook management
    # ------------------------------------------------------------------

    @prerelease_endpoint(key="webhooks.list", stage=ReleaseStage.ALPHA)
    def list(
        self,
        *,
        organization: str | None = None,
        name: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> ListWebhooksResponse:
        """List webhooks the user has access to.

        Webhooks are returned in descending creation order (most recently
        created first). Webhooks used as monitor notification channels are
        included.

        Args:
            organization: Optional organization ID or name to narrow the list
                to a single organization.
            name: Optional case-insensitive substring filter on the webhook
                name.
            limit: Maximum number of webhooks to return. The server enforces
                an upper bound.
            cursor: Opaque pagination cursor returned from a previous response.

        Returns:
            A response object with the webhooks and pagination information.

        Raises:
            NotFoundError: If the organization name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/429).
        """
        org_id = (
            _find_organization_id(self._organizations_api, organization)
            if organization is not None
            else None
        )
        return self._api.list_webhooks(
            org_id=org_id,
            name=name,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(key="webhooks.get", stage=ReleaseStage.ALPHA)
    def get(self, *, webhook: str, organization: str | None = None) -> Webhook:
        """Get a webhook by ID or name.

        Args:
            webhook: Webhook ID or name. If a name is provided, *organization*
                is required for resolution.
            organization: Organization ID or name. Required when *webhook* is
                a name so it can be resolved to an ID.

        Returns:
            The webhook object. Credentials (``auth_token``, header values,
            signing secret) are never included.

        Raises:
            NotFoundError: If the webhook name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/404/429).
        """
        webhook_id = _find_webhook_id(
            api=self._api,
            organizations_api=self._organizations_api,
            webhook=webhook,
            organization=organization,
        )
        return self._api.get_webhook(webhook_id=webhook_id)

    @prerelease_endpoint(key="webhooks.create", stage=ReleaseStage.ALPHA)
    def create(
        self,
        *,
        organization: str,
        name: str,
        url: str,
        description: str | None = None,
        auth_type: WebhookAuthType | None = None,
        auth_token: str | None = None,
        timeout_ms: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> CreateWebhookResponse:
        """Create a webhook.

        For ``HMAC_SHA256`` webhooks the response carries ``signing_secret``.
        This is the only time the secret is ever returned; store it securely.
        Afterwards only the redacted ``signing_secret_hint`` is readable, and
        losing the secret means deleting and recreating the webhook.

        Args:
            organization: Organization ID or name to create the webhook in.
            name: Webhook name (must be unique within the organization, max
                255 characters).
            url: The HTTPS endpoint events are delivered to.
            description: Optional description. Defaults to an empty string.
            auth_type: How deliveries are authenticated. Defaults to
                ``BEARER`` and cannot be changed after creation.
            auth_token: The complete ``Authorization`` header value sent with
                each delivery, e.g. ``"Bearer my-token"``. It is sent verbatim,
                so include the ``Bearer `` prefix if the endpoint expects one.
                Only valid when *auth_type* is ``BEARER``. Never returned.
            timeout_ms: Delivery timeout in milliseconds, between 1000 and
                60000. Defaults to 30000.
            headers: Custom HTTP headers sent with each delivery, at most 20.
                Header values are never returned.

        Returns:
            The created webhook, including ``signing_secret`` for
            ``HMAC_SHA256`` webhooks.

        Raises:
            NotFoundError: If the organization name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/409/422/429).
        """
        from arize._generated import api_client as gen

        org_id = _find_organization_id(self._organizations_api, organization)
        body = gen.CreateWebhookRequest(
            organization_id=org_id,
            name=name,
            url=url,
            description=description,
            auth_type=auth_type,
            auth_token=auth_token,
            timeout_ms=timeout_ms,
            headers=headers,
        )
        return self._api.create_webhook(create_webhook_request=body)

    @prerelease_endpoint(key="webhooks.update", stage=ReleaseStage.ALPHA)
    def update(
        self,
        *,
        webhook: str,
        organization: str | None = None,
        name: str | UNSET = _UNSET,
        description: str | None | UNSET = _UNSET,
        url: str | UNSET = _UNSET,
        auth_token: str | UNSET = _UNSET,
        timeout_ms: int | UNSET = _UNSET,
        headers: dict[str, str] | UNSET = _UNSET,
    ) -> Webhook:
        """Update a webhook.

        Only the fields passed are updated. At least one field must be
        provided. ``auth_type`` cannot be changed after creation, and the
        signing secret of an ``HMAC_SHA256`` webhook cannot be rotated; create
        a new webhook instead.

        Args:
            webhook: Webhook ID or name. If a name is provided, *organization*
                is required for resolution.
            organization: Organization ID or name. Required when *webhook* is
                a name so it can be resolved to an ID.
            name: New name (must remain unique within the organization).
            description: New description. Pass ``None`` to clear it.
            url: New HTTPS endpoint.
            auth_token: Replacement ``Authorization`` header value. Only valid
                when the webhook's ``auth_type`` is ``BEARER``.
            timeout_ms: New delivery timeout in milliseconds (1000 to 60000).
            headers: Replacement custom headers. Replaces the whole header
                map; headers not included are removed.

        Returns:
            The updated webhook object as returned by the API.

        Raises:
            ValueError: If no fields to update are provided.
            NotFoundError: If the webhook name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/409/422/429).
        """
        kwargs: dict[str, Any] = {}
        if is_provided(name):
            kwargs["name"] = name
        if is_provided(description):
            kwargs["description"] = description
        if is_provided(url):
            kwargs["url"] = url
        if is_provided(auth_token):
            kwargs["auth_token"] = auth_token
        if is_provided(timeout_ms):
            kwargs["timeout_ms"] = timeout_ms
        if is_provided(headers):
            kwargs["headers"] = headers
        if not kwargs:
            raise ValueError(
                "At least one of 'name', 'description', 'url', 'auth_token',"
                " 'timeout_ms', or 'headers' must be provided"
            )

        from arize._generated import api_client as gen

        webhook_id = _find_webhook_id(
            api=self._api,
            organizations_api=self._organizations_api,
            webhook=webhook,
            organization=organization,
        )
        body = gen.UpdateWebhookRequest(**kwargs)
        return self._api.update_webhook(
            webhook_id=webhook_id, update_webhook_request=body
        )

    @prerelease_endpoint(key="webhooks.delete", stage=ReleaseStage.ALPHA)
    def delete(self, *, webhook: str, organization: str | None = None) -> None:
        """Delete a webhook by ID or name.

        The webhook stops receiving events and is detached from every prompt,
        evaluator, and monitor it was subscribed to. This operation is
        irreversible.

        Args:
            webhook: Webhook ID or name. If a name is provided, *organization*
                is required for resolution.
            organization: Organization ID or name. Required when *webhook* is
                a name so it can be resolved to an ID.

        Returns:
            This method returns None on success (HTTP 204 No Content).

        Raises:
            NotFoundError: If the webhook name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/429).
        """
        webhook_id = _find_webhook_id(
            api=self._api,
            organizations_api=self._organizations_api,
            webhook=webhook,
            organization=organization,
        )
        return self._api.delete_webhook(webhook_id=webhook_id)

    @prerelease_endpoint(key="webhooks.test", stage=ReleaseStage.ALPHA)
    def test(
        self, *, webhook: str, organization: str | None = None
    ) -> TestWebhookResponse:
        """Send a test event to a webhook's endpoint and report the outcome.

        A successful call means the test ran; check ``status_code`` and
        ``error_message`` on the response for the endpoint's actual outcome.
        ``status_code`` is 502 when no response was received. Test deliveries
        are not supported for ``HMAC_SHA256`` webhooks.

        Args:
            webhook: Webhook ID or name. If a name is provided, *organization*
                is required for resolution.
            organization: Organization ID or name. Required when *webhook* is
                a name so it can be resolved to an ID.

        Returns:
            The endpoint's status code and error message, if any.

        Raises:
            NotFoundError: If the webhook name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/429/503).
        """
        webhook_id = _find_webhook_id(
            api=self._api,
            organizations_api=self._organizations_api,
            webhook=webhook,
            organization=organization,
        )
        return self._api.test_webhook(webhook_id=webhook_id)

    @prerelease_endpoint(
        key="webhooks.list_delivery_attempts", stage=ReleaseStage.ALPHA
    )
    def list_delivery_attempts(
        self,
        *,
        webhook: str,
        organization: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> ListWebhookDeliveryAttemptsResponse:
        """List a webhook's delivery attempts, most recent first.

        Each event may have several attempts, since failed deliveries are
        retried.

        Args:
            webhook: Webhook ID or name. If a name is provided, *organization*
                is required for resolution.
            organization: Organization ID or name. Required when *webhook* is
                a name so it can be resolved to an ID.
            limit: Maximum number of attempts to return. The server enforces
                an upper bound.
            cursor: Opaque pagination cursor returned from a previous response.

        Returns:
            A response object with the delivery attempts and pagination
            information.

        Raises:
            NotFoundError: If the webhook name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/429).
        """
        webhook_id = _find_webhook_id(
            api=self._api,
            organizations_api=self._organizations_api,
            webhook=webhook,
            organization=organization,
        )
        return self._api.list_webhook_delivery_attempts(
            webhook_id=webhook_id, limit=limit, cursor=cursor
        )

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    @prerelease_endpoint(
        key="webhooks.list_subscriptions", stage=ReleaseStage.ALPHA
    )
    def list_subscriptions(
        self,
        *,
        source_type: WebhookSourceType | None = None,
        source_id: str | None = None,
        limit: int = DEFAULT_LIST_LIMIT,
        cursor: str | None = None,
    ) -> ListWebhookSubscriptionsResponse:
        """List webhook subscriptions on prompts and evaluators.

        Subscriptions are returned in descending creation order. Each
        subscription delivers one event to one webhook, so a webhook that
        receives several events from a source appears once per event.

        Subscriptions whose webhook has since been deleted are dropped after
        the page is read, so a page may hold fewer than *limit* items while
        ``pagination.has_more`` is still true. Keep paging until it is false.

        Args:
            source_type: Restrict to one source kind. Must be given together
                with *source_id*.
            source_id: Restrict to one prompt or evaluator by ID. Must be
                given together with *source_type*.
            limit: Maximum number of subscriptions to return. The server
                enforces an upper bound.
            cursor: Opaque pagination cursor returned from a previous response.

        Returns:
            A response object with the subscriptions and pagination
            information.

        Raises:
            ValueError: If only one of *source_type* and *source_id* is given.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/429).
        """
        if (source_type is None) != (source_id is None):
            raise ValueError(
                "'source_type' and 'source_id' must be provided together"
            )
        return self._api.list_webhook_subscriptions(
            source_type=source_type,
            source_id=source_id,
            limit=limit,
            cursor=cursor,
        )

    @prerelease_endpoint(
        key="webhooks.create_subscription", stage=ReleaseStage.ALPHA
    )
    def create_subscription(
        self,
        *,
        webhook: str,
        source_type: WebhookSourceType,
        source_id: str,
        event: WebhookEventType,
        organization: str | None = None,
    ) -> WebhookSubscription:
        """Subscribe a webhook to one event on a prompt or evaluator.

        To deliver several events to the same webhook, create one subscription
        per event. The event must belong to the source type: prompt events for
        ``PROMPT`` sources and evaluator events for ``EVALUATOR`` sources.

        Args:
            webhook: Webhook ID or name. Must belong to the source's
                organization. If a name is provided, *organization* is
                required for resolution.
            source_type: The kind of resource to attach the webhook to.
            source_id: The ID of the prompt or evaluator.
            event: The event to deliver.
            organization: Organization ID or name. Required when *webhook* is
                a name so it can be resolved to an ID.

        Returns:
            The created subscription.

        Raises:
            NotFoundError: If the webhook name cannot be resolved.
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/409/422/429).
        """
        from arize._generated import api_client as gen

        webhook_id = _find_webhook_id(
            api=self._api,
            organizations_api=self._organizations_api,
            webhook=webhook,
            organization=organization,
        )
        body = gen.CreateWebhookSubscriptionRequest(
            webhook_id=webhook_id,
            source_type=source_type,
            source_id=source_id,
            event=event,
        )
        return self._api.create_webhook_subscription(
            create_webhook_subscription_request=body
        )

    @prerelease_endpoint(
        key="webhooks.get_subscription", stage=ReleaseStage.ALPHA
    )
    def get_subscription(self, *, subscription_id: str) -> WebhookSubscription:
        """Get a webhook subscription by ID.

        Args:
            subscription_id: The subscription ID.

        Returns:
            The subscription object.

        Raises:
            ApiException: If the REST API returns an error response
                (e.g. 400/401/404/429). A 404 is returned when the
                subscription does not exist, its source is not readable, or
                its webhook has since been deleted.
        """
        return self._api.get_webhook_subscription(
            subscription_id=subscription_id
        )

    @prerelease_endpoint(
        key="webhooks.delete_subscription", stage=ReleaseStage.ALPHA
    )
    def delete_subscription(self, *, subscription_id: str) -> None:
        """Delete a webhook subscription by ID.

        The webhook stops receiving that event from the source. Other
        subscriptions on the source and the webhook itself are unaffected.

        Args:
            subscription_id: The subscription ID.

        Returns:
            This method returns None on success (HTTP 204 No Content).

        Raises:
            ApiException: If the REST API returns an error response
                (e.g. 400/401/403/404/429).
        """
        return self._api.delete_webhook_subscription(
            subscription_id=subscription_id
        )
