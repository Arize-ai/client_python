# arize._generated.api_client.APIKeysApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**api_keys_create**](APIKeysApi.md#api_keys_create) | **POST** /v2/api-keys | Create an API key
[**api_keys_delete**](APIKeysApi.md#api_keys_delete) | **DELETE** /v2/api-keys/{api_key_id} | Delete an API key
[**api_keys_list**](APIKeysApi.md#api_keys_list) | **GET** /v2/api-keys | List API keys
[**api_keys_refresh**](APIKeysApi.md#api_keys_refresh) | **POST** /v2/api-keys/{api_key_id}/refresh | Refresh an API key


# **api_keys_create**
> ApiKeyCreated api_keys_create(api_key_create)

Create an API key

Create a new API key for the authenticated user.

- `key_type` defaults to `user` when omitted.
- For `service` keys, `space_id` is required. The service key is
  scoped to the given space, and a bot user will be created with the specified roles.
- For `user` keys, `space_id` and `roles` must not be set — passing either returns `400`.
  The key inherits the authenticated user's own permissions.
- You may only assign roles at or below your own privilege level. Attempting to
  assign a role higher than your own returns `400 Bad Request`.
- All roles default to the minimum privilege when omitted: `space_role` → `member`,
  `org_role` → `read-only`, `account_role` → `member`.

The full API key value (`key`) is **only returned once** in the creation response.
Store it securely — it cannot be retrieved again. Use the `redacted_key` field on
subsequent reads.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.api_key_create import ApiKeyCreate
from arize._generated.api_client.models.api_key_created import ApiKeyCreated
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
    api_instance = arize._generated.api_client.APIKeysApi(api_client)
    api_key_create = {"name":"CI pipeline key"} # ApiKeyCreate | Body containing API key creation parameters

    try:
        # Create an API key
        api_response = api_instance.api_keys_create(api_key_create)
        print("The response of APIKeysApi->api_keys_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling APIKeysApi->api_keys_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **api_key_create** | [**ApiKeyCreate**](ApiKeyCreate.md)| Body containing API key creation parameters | 

### Return type

[**ApiKeyCreated**](ApiKeyCreated.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | API key successfully created or refreshed. The raw key is only returned once. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Invalid request |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **api_keys_delete**
> api_keys_delete(api_key_id)

Delete an API key

Delete an API key by its ID (soft-delete). This operation is irreversible. The key will
immediately stop working for authentication.

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
    api_instance = arize._generated.api_client.APIKeysApi(api_client)
    api_key_id = 'QXBpS2V5OjEyMzQ1' # str | The unique identifier of the API key

    try:
        # Delete an API key
        api_instance.api_keys_delete(api_key_id)
    except Exception as e:
        print("Exception when calling APIKeysApi->api_keys_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **api_key_id** | **str**| The unique identifier of the API key | 

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
**204** | API key successfully deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **api_keys_list**
> ApiKeysList200Response api_keys_list(key_type=key_type, status=status, limit=limit, cursor=cursor)

List API keys

List API keys for the authenticated user. Returns metadata for each key (id, name, description,
key_type, status, redacted_key, created_at, expires_at, created_by_user_id). The raw key
secret is never returned after creation.

Results can be filtered by key type, status, and created-by user ID. Responses are
paginated; use `limit` and `cursor` and the response `pagination.next_cursor` for
subsequent pages.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.api_key_status import ApiKeyStatus
from arize._generated.api_client.models.api_keys_list200_response import ApiKeysList200Response
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
    api_instance = arize._generated.api_client.APIKeysApi(api_client)
    key_type = 'user' # str | Filter by API key type. - user - Key associated with a specific user. - service - Key associated with a bot user for service authentication.  (optional)
    status = arize._generated.api_client.ApiKeyStatus() # ApiKeyStatus | Filter by API key status. - active - Only return keys that are valid for use. - deleted - Only return keys that have been deleted.  When not specified, defaults to `active`.  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List API keys
        api_response = api_instance.api_keys_list(key_type=key_type, status=status, limit=limit, cursor=cursor)
        print("The response of APIKeysApi->api_keys_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling APIKeysApi->api_keys_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **key_type** | **str**| Filter by API key type. - user - Key associated with a specific user. - service - Key associated with a bot user for service authentication.  | [optional] 
 **status** | [**ApiKeyStatus**](.md)| Filter by API key status. - active - Only return keys that are valid for use. - deleted - Only return keys that have been deleted.  When not specified, defaults to &#x60;active&#x60;.  | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ApiKeysList200Response**](ApiKeysList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of API keys for the authenticated user. The raw key secret is never returned. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**422** | Invalid request |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **api_keys_refresh**
> ApiKeyCreated api_keys_refresh(api_key_id, api_key_refresh=api_key_refresh)

Refresh an API key

Atomically revoke an existing API key and issue a replacement with the same
metadata (name, description, and key type). Attempting to refresh a key
you did not create returns a `400` error.

The old key is invalidated and the new key is activated in a single transaction —
there is no window where neither key is valid. The full new key value (`key`) is
**only returned once** in the response. Store it securely.

**Expiry behaviour:** Supply `expires_at` in the request body to set an expiration
on the replacement key. Omit `expires_at` (or send an empty body `{}`) to create
the replacement key with no expiration (infinite lifetime).

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.api_key_created import ApiKeyCreated
from arize._generated.api_client.models.api_key_refresh import ApiKeyRefresh
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
    api_instance = arize._generated.api_client.APIKeysApi(api_client)
    api_key_id = 'QXBpS2V5OjEyMzQ1' # str | The unique identifier of the API key
    api_key_refresh = {} # ApiKeyRefresh | Optional body for overriding expiry on a refreshed API key. (optional)

    try:
        # Refresh an API key
        api_response = api_instance.api_keys_refresh(api_key_id, api_key_refresh=api_key_refresh)
        print("The response of APIKeysApi->api_keys_refresh:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling APIKeysApi->api_keys_refresh: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **api_key_id** | **str**| The unique identifier of the API key | 
 **api_key_refresh** | [**ApiKeyRefresh**](ApiKeyRefresh.md)| Optional body for overriding expiry on a refreshed API key. | [optional] 

### Return type

[**ApiKeyCreated**](ApiKeyCreated.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | API key successfully created or refreshed. The raw key is only returned once. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

