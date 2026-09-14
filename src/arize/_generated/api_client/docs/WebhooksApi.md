# arize._generated.api_client.WebhooksApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_webhook**](WebhooksApi.md#create_webhook) | **POST** /v2/webhooks | Create a webhook
[**create_webhook_subscription**](WebhooksApi.md#create_webhook_subscription) | **POST** /v2/webhook-subscriptions | Create a webhook subscription
[**delete_webhook**](WebhooksApi.md#delete_webhook) | **DELETE** /v2/webhooks/{webhook_id} | Delete a webhook
[**delete_webhook_subscription**](WebhooksApi.md#delete_webhook_subscription) | **DELETE** /v2/webhook-subscriptions/{subscription_id} | Delete a webhook subscription
[**get_webhook**](WebhooksApi.md#get_webhook) | **GET** /v2/webhooks/{webhook_id} | Get a webhook
[**get_webhook_subscription**](WebhooksApi.md#get_webhook_subscription) | **GET** /v2/webhook-subscriptions/{subscription_id} | Get a webhook subscription
[**list_webhook_delivery_attempts**](WebhooksApi.md#list_webhook_delivery_attempts) | **GET** /v2/webhooks/{webhook_id}/delivery-attempts | List a webhook&#39;s delivery attempts
[**list_webhook_subscriptions**](WebhooksApi.md#list_webhook_subscriptions) | **GET** /v2/webhook-subscriptions | List webhook subscriptions
[**list_webhooks**](WebhooksApi.md#list_webhooks) | **GET** /v2/webhooks | List webhooks
[**test_webhook**](WebhooksApi.md#test_webhook) | **POST** /v2/webhooks/{webhook_id}/test | Send a test event to a webhook
[**update_webhook**](WebhooksApi.md#update_webhook) | **PATCH** /v2/webhooks/{webhook_id} | Update a webhook


# **create_webhook**
> CreateWebhookResponse create_webhook(create_webhook_request)

Create a webhook

Create a new webhook in an organization.

**Payload Requirements**
- `organization_id`, `name`, and `url` are required.
- The webhook name must be unique within the organization (409 on conflict).
- `auth_type` is optional, defaults to `BEARER`, and cannot be changed
  after creation.
- `auth_token` is only valid when `auth_type` is `BEARER`, and is
  write-only — it is never returned in any response.
- `timeout_ms` is optional, defaults to 30000, and must be between
  1000 and 60000.
- `headers` is optional and holds at most 20 entries; header names
  must be valid HTTP header names, and connection-management headers
  are rejected. Write-only: header values are never returned.
- System-managed fields (`id`, `created_at`, `updated_at`) are
  generated automatically and rejected if provided.

For `HMAC_SHA256` webhooks, a signing secret is generated and returned
in this response — **the only time it is ever returned**. Store it
securely: only a redacted hint is readable afterwards, and losing the
secret means deleting and recreating the webhook.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_webhook_request import CreateWebhookRequest
from arize._generated.api_client.models.create_webhook_response import CreateWebhookResponse
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.WebhooksApi(api_client)
    create_webhook_request = {"organization_id":"T3JnYW5pemF0aW9uOjEyMzQ1","name":"Prompt release notifications","url":"https://example.com/hooks/arize","description":"Notifies the deploy pipeline when a prompt version is labeled","auth_type":"HMAC_SHA256","timeout_ms":30000,"headers":{"X-Environment":"production"}} # CreateWebhookRequest | Body containing webhook creation parameters

    try:
        # Create a webhook
        api_response = api_instance.create_webhook(create_webhook_request)
        print("The response of WebhooksApi->create_webhook:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling WebhooksApi->create_webhook: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_webhook_request** | [**CreateWebhookRequest**](CreateWebhookRequest.md)| Body containing webhook creation parameters | 

### Return type

[**CreateWebhookResponse**](CreateWebhookResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | The created webhook. For &#x60;HMAC_SHA256&#x60; webhooks the response includes &#x60;signing_secret&#x60; — the only time it is ever returned.  |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **create_webhook_subscription**
> WebhookSubscription create_webhook_subscription(create_webhook_subscription_request)

Create a webhook subscription

Subscribe a webhook to one event on a prompt or evaluator. To deliver
several events to the same webhook, create one subscription per event.

**Payload Requirements**
- `webhook_id`, `source_type`, `source_id`, and `event` are required.
- `webhook_id` must be a webhook in the source's organization; unknown
  webhooks yield a 404.
- `event` must belong to the source type: prompt events for `PROMPT`
  sources and evaluator events for `EVALUATOR` sources.
  Other combinations are rejected with a 422.
- A webhook can subscribe to a given event on a given source only once
  (409 on conflict).
- At most 200 webhooks may subscribe to the same event on a source;
  requests that would exceed this limit are rejected with a 422.

Creating a subscription requires `PROMPT_UPDATE` for a prompt source or
`EVALUATOR_UPDATE` for an evaluator source. Callers with the matching
`PROMPT_READ` or `EVALUATOR_READ` permission but not the required update
permission receive a 403; sources the caller cannot read yield a 404.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_webhook_subscription_request import CreateWebhookSubscriptionRequest
from arize._generated.api_client.models.webhook_subscription import WebhookSubscription
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.WebhooksApi(api_client)
    create_webhook_subscription_request = {"webhook_id":"V2ViaG9vazoxMjM0NQ==","source_type":"PROMPT","source_id":"UHJvbXB0OjEyMzQ1","event":"PROMPT_VERSION_LABELED"} # CreateWebhookSubscriptionRequest | Body containing the webhook, the source to attach it to, and the event to deliver

    try:
        # Create a webhook subscription
        api_response = api_instance.create_webhook_subscription(create_webhook_subscription_request)
        print("The response of WebhooksApi->create_webhook_subscription:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling WebhooksApi->create_webhook_subscription: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_webhook_subscription_request** | [**CreateWebhookSubscriptionRequest**](CreateWebhookSubscriptionRequest.md)| Body containing the webhook, the source to attach it to, and the event to deliver | 

### Return type

[**WebhookSubscription**](WebhookSubscription.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | The created webhook subscription |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_webhook**
> delete_webhook(webhook_id)

Delete a webhook

Delete a webhook by its ID. The webhook stops receiving events and is
detached from every prompt, evaluator, and monitor it was subscribed
to. This operation is irreversible.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.WebhooksApi(api_client)
    webhook_id = 'V2ViaG9vazoxMjM0NQ==' # str | The unique webhook identifier (base64)

    try:
        # Delete a webhook
        api_instance.delete_webhook(webhook_id)
    except Exception as e:
        print("Exception when calling WebhooksApi->delete_webhook: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **webhook_id** | **str**| The unique webhook identifier (base64) | 

### Return type

void (empty response body)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**204** | Webhook successfully deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_webhook_subscription**
> delete_webhook_subscription(subscription_id)

Delete a webhook subscription

Delete a webhook subscription by its ID. The webhook stops receiving
that event from the source. Other subscriptions on the source and the
webhook itself are unaffected.

Deleting a subscription requires `PROMPT_UPDATE` for a prompt source or
`EVALUATOR_UPDATE` for an evaluator source. Callers with the matching
`PROMPT_READ` or `EVALUATOR_READ` permission but not the required update
permission receive a 403; sources the caller cannot read yield a 404.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.WebhooksApi(api_client)
    subscription_id = 'V2ViaG9va1N1YnNjcmlwdGlvbjoxMjM0NQ==' # str | The unique webhook subscription identifier (base64)

    try:
        # Delete a webhook subscription
        api_instance.delete_webhook_subscription(subscription_id)
    except Exception as e:
        print("Exception when calling WebhooksApi->delete_webhook_subscription: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **subscription_id** | **str**| The unique webhook subscription identifier (base64) | 

### Return type

void (empty response body)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**204** | Webhook subscription successfully deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_webhook**
> Webhook get_webhook(webhook_id)

Get a webhook

Get a specific webhook by its ID.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.webhook import Webhook
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.WebhooksApi(api_client)
    webhook_id = 'V2ViaG9vazoxMjM0NQ==' # str | The unique webhook identifier (base64)

    try:
        # Get a webhook
        api_response = api_instance.get_webhook(webhook_id)
        print("The response of WebhooksApi->get_webhook:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling WebhooksApi->get_webhook: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **webhook_id** | **str**| The unique webhook identifier (base64) | 

### Return type

[**Webhook**](Webhook.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A webhook object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_webhook_subscription**
> WebhookSubscription get_webhook_subscription(subscription_id)

Get a webhook subscription

Get a specific webhook subscription by its ID. A 404 is returned when
the subscription does not exist, its source is not readable, or its
webhook has since been deleted.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.webhook_subscription import WebhookSubscription
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.WebhooksApi(api_client)
    subscription_id = 'V2ViaG9va1N1YnNjcmlwdGlvbjoxMjM0NQ==' # str | The unique webhook subscription identifier (base64)

    try:
        # Get a webhook subscription
        api_response = api_instance.get_webhook_subscription(subscription_id)
        print("The response of WebhooksApi->get_webhook_subscription:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling WebhooksApi->get_webhook_subscription: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **subscription_id** | **str**| The unique webhook subscription identifier (base64) | 

### Return type

[**WebhookSubscription**](WebhookSubscription.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A webhook subscription object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_webhook_delivery_attempts**
> ListWebhookDeliveryAttemptsResponse list_webhook_delivery_attempts(webhook_id, limit=limit, cursor=cursor)

List a webhook's delivery attempts

List the webhook's delivery attempts, most recent first. Each event
may have several attempts, since failed deliveries are retried.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_webhook_delivery_attempts_response import ListWebhookDeliveryAttemptsResponse
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.WebhooksApi(api_client)
    webhook_id = 'V2ViaG9vazoxMjM0NQ==' # str | The unique webhook identifier (base64)
    limit = 50 # int | Maximum items to return. Defaults to 50 if omitted; maximum is 500. (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List a webhook's delivery attempts
        api_response = api_instance.list_webhook_delivery_attempts(webhook_id, limit=limit, cursor=cursor)
        print("The response of WebhooksApi->list_webhook_delivery_attempts:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling WebhooksApi->list_webhook_delivery_attempts: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **webhook_id** | **str**| The unique webhook identifier (base64) | 
 **limit** | **int**| Maximum items to return. Defaults to 50 if omitted; maximum is 500. | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListWebhookDeliveryAttemptsResponse**](ListWebhookDeliveryAttemptsResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of delivery attempts, most recent first |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_webhook_subscriptions**
> ListWebhookSubscriptionsResponse list_webhook_subscriptions(source_type=source_type, source_id=source_id, limit=limit, cursor=cursor)

List webhook subscriptions

List webhook subscriptions attached to prompts and evaluators the caller
can read, most recently created first. To list one source, provide both
`source_type` and `source_id`; providing only one returns a 400. An
unfiltered list returns a 403 when the caller can read no prompt or
evaluator at all.

Each subscription delivers one event to one webhook, so a webhook that
receives several events from the source appears once per event.
Subscriptions whose webhook has since been deleted are omitted after
the page is read, so a page may hold fewer than `limit` items, or none,
while `has_more` is still `true`. Keep paging until `has_more` is
`false`.

When filtering by source, a 404 is returned when the source does not
exist or is not readable.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_webhook_subscriptions_response import ListWebhookSubscriptionsResponse
from arize._generated.api_client.models.webhook_source_type import WebhookSourceType
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.WebhooksApi(api_client)
    source_type = arize._generated.api_client.WebhookSourceType() # WebhookSourceType | Filter subscriptions to one kind of source. Must be paired with `source_id`. When both filters are omitted, subscriptions from every supported source type are returned.  (optional)
    source_id = 'UHJvbXB0OjEyMzQ1' # str | Filter subscriptions to one prompt or evaluator. Must be paired with `source_type`. When both filters are omitted, subscriptions from every readable supported source are returned.  (optional)
    limit = 50 # int | Maximum items to return. Defaults to 50 if omitted; maximum is 100. (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List webhook subscriptions
        api_response = api_instance.list_webhook_subscriptions(source_type=source_type, source_id=source_id, limit=limit, cursor=cursor)
        print("The response of WebhooksApi->list_webhook_subscriptions:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling WebhooksApi->list_webhook_subscriptions: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **source_type** | [**WebhookSourceType**](.md)| Filter subscriptions to one kind of source. Must be paired with &#x60;source_id&#x60;. When both filters are omitted, subscriptions from every supported source type are returned.  | [optional] 
 **source_id** | **str**| Filter subscriptions to one prompt or evaluator. Must be paired with &#x60;source_type&#x60;. When both filters are omitted, subscriptions from every readable supported source are returned.  | [optional] 
 **limit** | **int**| Maximum items to return. Defaults to 50 if omitted; maximum is 100. | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListWebhookSubscriptionsResponse**](ListWebhookSubscriptionsResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of webhook subscriptions, most recently created first |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_webhooks**
> ListWebhooksResponse list_webhooks(org_id=org_id, name=name, limit=limit, cursor=cursor)

List webhooks

List the webhooks in the organizations the user has access to, most
recently created first. Use `org_id` to narrow the list to a single
organization; a 404 is returned only when the given `org_id` does not
exist or is not accessible.

Webhooks used as monitor notification channels are included — a
webhook is an organization-level destination regardless of what it is
attached to.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_webhooks_response import ListWebhooksResponse
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.WebhooksApi(api_client)
    org_id = 'T3JnYW5pemF0aW9uOjEyMzQ1' # str | The unique organization identifier (base64). When provided, only resources belonging to this organization are returned. (optional)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    limit = 50 # int | Maximum items to return. Defaults to 50 if omitted; maximum is 100. (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List webhooks
        api_response = api_instance.list_webhooks(org_id=org_id, name=name, limit=limit, cursor=cursor)
        print("The response of WebhooksApi->list_webhooks:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling WebhooksApi->list_webhooks: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **org_id** | **str**| The unique organization identifier (base64). When provided, only resources belonging to this organization are returned. | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. If omitted, no name filtering is applied and all resources are returned.  | [optional] 
 **limit** | **int**| Maximum items to return. Defaults to 50 if omitted; maximum is 100. | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListWebhooksResponse**](ListWebhooksResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of webhook objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **test_webhook**
> TestWebhookResponse test_webhook(webhook_id)

Send a test event to a webhook

Send a test event to the webhook's endpoint and report the outcome.
Use this to verify the endpoint is reachable and accepts deliveries
before subscribing the webhook to real events.

A 200 response means the test ran — check `status_code` and
`error_message` in the body for the endpoint's actual outcome.
A 404 means the destination was not available to receive a test.
A 503 means the test could not be sent; retry later.

Test deliveries are not supported for `HMAC_SHA256` webhooks; those
requests fail with a 400.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.test_webhook_response import TestWebhookResponse
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.WebhooksApi(api_client)
    webhook_id = 'V2ViaG9vazoxMjM0NQ==' # str | The unique webhook identifier (base64)

    try:
        # Send a test event to a webhook
        api_response = api_instance.test_webhook(webhook_id)
        print("The response of WebhooksApi->test_webhook:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling WebhooksApi->test_webhook: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **webhook_id** | **str**| The unique webhook identifier (base64) | 

### Return type

[**TestWebhookResponse**](TestWebhookResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | The outcome of the test delivery |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |
**503** | Service temporarily unavailable |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_webhook**
> Webhook update_webhook(webhook_id, update_webhook_request)

Update a webhook

Update a webhook by its ID. At least one field must be provided.

**Payload Requirements**
- At least one of `name`, `description`, `url`, `auth_token`,
  `timeout_ms`, or `headers` must be provided.
- If `name` is provided, it must be unique within the organization
  (409 on conflict).
- `headers` replaces the whole header map. Write-only: header values
  are never returned.
- `auth_type` cannot be changed after creation, and the signing secret
  of an `HMAC_SHA256` webhook cannot be rotated — create a new webhook
  instead.
- System-managed fields (`id`, `created_at`, `updated_at`) cannot be
  modified.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.update_webhook_request import UpdateWebhookRequest
from arize._generated.api_client.models.webhook import Webhook
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.WebhooksApi(api_client)
    webhook_id = 'V2ViaG9vazoxMjM0NQ==' # str | The unique webhook identifier (base64)
    update_webhook_request = {"name":"Prompt release notifications (staging)","timeout_ms":10000} # UpdateWebhookRequest | Body containing webhook update parameters. At least one field must be provided.

    try:
        # Update a webhook
        api_response = api_instance.update_webhook(webhook_id, update_webhook_request)
        print("The response of WebhooksApi->update_webhook:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling WebhooksApi->update_webhook: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **webhook_id** | **str**| The unique webhook identifier (base64) | 
 **update_webhook_request** | [**UpdateWebhookRequest**](UpdateWebhookRequest.md)| Body containing webhook update parameters. At least one field must be provided. | 

### Return type

[**Webhook**](Webhook.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A webhook object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

