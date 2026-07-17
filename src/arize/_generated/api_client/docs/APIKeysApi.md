# arize._generated.api_client.APIKeysApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_api_key**](APIKeysApi.md#create_api_key) | **POST** /v2/api-keys | Create an API key
[**list_api_keys**](APIKeysApi.md#list_api_keys) | **GET** /v2/api-keys | List API keys
[**refresh_api_key**](APIKeysApi.md#refresh_api_key) | **POST** /v2/api-keys/{api_key_id}/refresh | Refresh an API key
[**revoke_api_key**](APIKeysApi.md#revoke_api_key) | **PUT** /v2/api-keys/{api_key_id}/revoke | Revoke an API key


# **create_api_key**
> ApiKey create_api_key(create_api_key_request)

Create an API key

Create a new API key for the authenticated user.

- `key_type` defaults to `USER` when omitted.
- For `SERVICE` keys, `space_id` is required. The service key is
  scoped to the given space, and a bot user will be created with the specified roles.
- For `USER` keys, `space_id` and `roles` must not be set — passing either returns `400`.
  The key inherits the authenticated user's own permissions.
- You may only assign roles at or below your own privilege level. Attempting to
  assign a role higher than your own returns `400 Bad Request`.
- All roles default to the minimum privilege when omitted: `space_role` → `MEMBER`,
  `org_role` → `READ_ONLY`, `account_role` → `MEMBER`.

**Authorization:**
- **User keys:** Requires the `developer` user permission flag. Returns `403` when this flag is absent.
- **Service keys:** Requires the `SERVICE_KEY_CREATE` permission in the target space (space
  member or above).

The full API key value (`key`) is **only returned once** in the creation response.
Store it securely — it cannot be retrieved again. Use the `redacted_key` field on
subsequent reads.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.api_key import ApiKey
from arize._generated.api_client.models.create_api_key_request import CreateApiKeyRequest
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
    create_api_key_request = {"name":"CI pipeline key"} # CreateApiKeyRequest | Body containing API key creation parameters

    try:
        # Create an API key
        api_response = api_instance.create_api_key(create_api_key_request)
        print("The response of APIKeysApi->create_api_key:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling APIKeysApi->create_api_key: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_api_key_request** | [**CreateApiKeyRequest**](CreateApiKeyRequest.md)| Body containing API key creation parameters | 

### Return type

[**ApiKey**](ApiKey.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | API key successfully created. The raw key value is only returned once — store it securely. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_api_keys**
> ListApiKeysResponse list_api_keys(key_type=key_type, status=status, space_id=space_id, user_id=user_id, limit=limit, cursor=cursor)

List API keys

List API keys. Returns metadata for each key (id, name, description,
key_type, status, redacted_key, created_at, expires_at, created_by_user_id). The raw key
secret is never returned after creation.

Results can be filtered by key type, status, space, and creator. Responses are
paginated; use `limit` and `cursor` and the response `pagination.next_cursor` for
subsequent pages.

**Service keys (`key_type=SERVICE`):** Provide `space_id` to return all service keys for
that space. When `key_type` is omitted alongside `space_id`, service keys are returned
implicitly. Requires the `SERVICE_KEY_READ` permission in the space (or account/space admin).
Optionally combine with `user_id` to filter service keys by their creator — available to any
caller with space access (not admin-gated).

**User keys (`key_type=USER`):** Returned by default (no `space_id`). Provide `user_id` to
view keys belonging to a specific user — account admins only; non-admins receive `403`.

**Authorization:** Requires the `developer` user permission flag or account admin role.
Returns `403` when neither condition is met.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.api_key_status import ApiKeyStatus
from arize._generated.api_client.models.api_key_type import ApiKeyType
from arize._generated.api_client.models.list_api_keys_response import ListApiKeysResponse
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
    key_type = arize._generated.api_client.ApiKeyType() # ApiKeyType | Filter by API key type. - USER - Key associated with a specific user. - SERVICE - Key associated with a bot user for service authentication.  (optional)
    status = arize._generated.api_client.ApiKeyStatus() # ApiKeyStatus | Filter by API key status. - ACTIVE - Only return keys that are valid for use. - REVOKED - Only return keys that have been revoked.  When not specified, defaults to `ACTIVE`.  (optional)
    space_id = 'U3BhY2U6MTIzNDU=' # str | Filter search results to a particular space ID (optional)
    user_id = 'VXNlcjoxMjM0NQ==' # str | Filter results by user (base64 global user ID). When provided, only records associated with this user are returned. Access requirements vary by endpoint — some endpoints restrict this filter to account admins.  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List API keys
        api_response = api_instance.list_api_keys(key_type=key_type, status=status, space_id=space_id, user_id=user_id, limit=limit, cursor=cursor)
        print("The response of APIKeysApi->list_api_keys:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling APIKeysApi->list_api_keys: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **key_type** | [**ApiKeyType**](.md)| Filter by API key type. - USER - Key associated with a specific user. - SERVICE - Key associated with a bot user for service authentication.  | [optional] 
 **status** | [**ApiKeyStatus**](.md)| Filter by API key status. - ACTIVE - Only return keys that are valid for use. - REVOKED - Only return keys that have been revoked.  When not specified, defaults to &#x60;ACTIVE&#x60;.  | [optional] 
 **space_id** | **str**| Filter search results to a particular space ID | [optional] 
 **user_id** | **str**| Filter results by user (base64 global user ID). When provided, only records associated with this user are returned. Access requirements vary by endpoint — some endpoints restrict this filter to account admins.  | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListApiKeysResponse**](ListApiKeysResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of API keys matching the request filters. The raw key secret is never returned. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **refresh_api_key**
> ApiKey refresh_api_key(api_key_id, refresh_api_key_request=refresh_api_key_request)

Refresh an API key

Atomically revoke an existing API key and issue a replacement with the same
metadata (name, description, and key type).

The old key is invalidated and the new key is activated in a single transaction —
there is no window where neither key is valid. The full new key value (`key`) is
**only returned once** in the response. Store it securely.

**Authorization:**
- **User keys:** the creator or an account admin may refresh the key. Requires the
  `developer` user permission flag. Returns `403` when this flag is absent.
- **Service keys:** space admins (and higher) may refresh any service key in their space.
  Non-admins require the `SERVICE_KEY_CREATE` permission and must be the creator of the key.

**Expiry behaviour:** `expires_at` is **required** when the existing key has an expiry
— omitting it would extend the key's lifetime to unbounded and is rejected with `422`.
For unbounded existing keys, `expires_at` may be omitted (the replacement is also
unbounded) or supplied to add a specific expiry. The value must not be later than the
existing key's expiry; to issue a key with a longer lifetime, use `POST /v2/api-keys`.

**Grace period:** Supply `grace_period_seconds` in the request body to keep the old key
valid for that many seconds after the refresh. If not supplied, the old key is revoked immediately.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.api_key import ApiKey
from arize._generated.api_client.models.refresh_api_key_request import RefreshApiKeyRequest
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
    api_key_id = 'QXBpS2V5OjEyMzQ1' # str | The unique API key identifier (base64)
    refresh_api_key_request = {} # RefreshApiKeyRequest | Optional body for tightening expiry on the new key and/or setting a grace period on the old key. Refresh cannot extend a key's lifetime: with an empty body the refreshed key inherits the old key's expiry, and an explicit `expires_at` later than the old key's expiry is rejected with 422.  (optional)

    try:
        # Refresh an API key
        api_response = api_instance.refresh_api_key(api_key_id, refresh_api_key_request=refresh_api_key_request)
        print("The response of APIKeysApi->refresh_api_key:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling APIKeysApi->refresh_api_key: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **api_key_id** | **str**| The unique API key identifier (base64) | 
 **refresh_api_key_request** | [**RefreshApiKeyRequest**](RefreshApiKeyRequest.md)| Optional body for tightening expiry on the new key and/or setting a grace period on the old key. Refresh cannot extend a key&#39;s lifetime: with an empty body the refreshed key inherits the old key&#39;s expiry, and an explicit &#x60;expires_at&#x60; later than the old key&#39;s expiry is rejected with 422.  | [optional] 

### Return type

[**ApiKey**](ApiKey.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | API key successfully refreshed. The raw value of the new key is only returned once — store it securely. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **revoke_api_key**
> revoke_api_key(api_key_id)

Revoke an API key

Revoke an API key by its ID. The key will immediately stop working for authentication. Revoking an
already-revoked key is a no-op and still returns `204`.

**Authorization:** 
Requires the `developer` user permission flag and account admin role. Returns `403` when conditions are not met.

  <Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


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
    api_key_id = 'QXBpS2V5OjEyMzQ1' # str | The unique API key identifier (base64)

    try:
        # Revoke an API key
        api_instance.revoke_api_key(api_key_id)
    except Exception as e:
        print("Exception when calling APIKeysApi->revoke_api_key: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **api_key_id** | **str**| The unique API key identifier (base64) | 

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
**204** | API key successfully revoked |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

