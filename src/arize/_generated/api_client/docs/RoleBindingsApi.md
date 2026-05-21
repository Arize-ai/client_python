# arize._generated.api_client.RoleBindingsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**role_bindings_create**](RoleBindingsApi.md#role_bindings_create) | **POST** /v2/role-bindings | Create a role binding
[**role_bindings_delete**](RoleBindingsApi.md#role_bindings_delete) | **DELETE** /v2/role-bindings/{binding_id} | Delete a role binding
[**role_bindings_get**](RoleBindingsApi.md#role_bindings_get) | **GET** /v2/role-bindings/{binding_id} | Get a role binding
[**role_bindings_list**](RoleBindingsApi.md#role_bindings_list) | **GET** /v2/role-bindings | List role bindings
[**role_bindings_update**](RoleBindingsApi.md#role_bindings_update) | **PATCH** /v2/role-bindings/{binding_id} | Update a role binding


# **role_bindings_create**
> RoleBinding role_bindings_create(role_binding_create)

Create a role binding

Create a new role binding that assigns a role to a user on a resource.

**Payload Requirements**
- `role_id`, `user_id`, `resource_type`, and `resource_id` are required.
- `resource_type` must be `SPACE` or `PROJECT`.
- `resource_id` must be a unique identifier for the selected `resource_type`.
- Only one binding per user and resource is allowed. If the target user
  already has any binding on the resource, the request returns
  `409 Conflict`.
- System-managed fields (`id`, `created_at`, `updated_at`) are returned
  by the server and are rejected on input.

**Valid example**
```json
{
  "role_id": "Um9sZToxOlY0S2E=",
  "user_id": "VXNlcjoxOmxQZzI=",
  "resource_type": "PROJECT",
  "resource_id": "TW9kZWw6MTpGdmxM"
}
```

**Invalid example**
```json
{
  "role_id": "Um9sZToxOlY0S2E=",
  "user_id": "VXNlcjoxOmxQZzI=",
  "resource_type": "PROJECT",
  "resource_id": "U3BhY2U6MTp1Rk4x"
}
```
This fails because `resource_id` must encode a `PROJECT` ID when
`resource_type` is `PROJECT`.

Use `PATCH /v2/role-bindings/{binding_id}` to change the assigned role
for an existing binding.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.role_binding import RoleBinding
from arize._generated.api_client.models.role_binding_create import RoleBindingCreate
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
    api_instance = arize._generated.api_client.RoleBindingsApi(api_client)
    role_binding_create = {"role_id":"Um9sZToxOmFCY0Q=","user_id":"VXNlcjo0MjphQmNE","resource_type":"SPACE","resource_id":"U3BhY2U6MTpWNEth"} # RoleBindingCreate | Body containing role binding creation parameters.

    try:
        # Create a role binding
        api_response = api_instance.role_bindings_create(role_binding_create)
        print("The response of RoleBindingsApi->role_bindings_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling RoleBindingsApi->role_bindings_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **role_binding_create** | [**RoleBindingCreate**](RoleBindingCreate.md)| Body containing role binding creation parameters. | 

### Return type

[**RoleBinding**](RoleBinding.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Role binding successfully created. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **role_bindings_delete**
> role_bindings_delete(binding_id)

Delete a role binding

Delete a role binding by its ID.

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
    api_instance = arize._generated.api_client.RoleBindingsApi(api_client)
    binding_id = 'Um9sZUJpbmRpbmc6MTphQmNE' # str | The unique role binding identifier (base64)

    try:
        # Delete a role binding
        api_instance.role_bindings_delete(binding_id)
    except Exception as e:
        print("Exception when calling RoleBindingsApi->role_bindings_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **binding_id** | **str**| The unique role binding identifier (base64) | 

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
**204** | Role binding successfully deleted. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **role_bindings_get**
> RoleBinding role_bindings_get(binding_id)

Get a role binding

Get a single role binding by its ID.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.role_binding import RoleBinding
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
    api_instance = arize._generated.api_client.RoleBindingsApi(api_client)
    binding_id = 'Um9sZUJpbmRpbmc6MTphQmNE' # str | The unique role binding identifier (base64)

    try:
        # Get a role binding
        api_response = api_instance.role_bindings_get(binding_id)
        print("The response of RoleBindingsApi->role_bindings_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling RoleBindingsApi->role_bindings_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **binding_id** | **str**| The unique role binding identifier (base64) | 

### Return type

[**RoleBinding**](RoleBinding.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A role binding object. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **role_bindings_list**
> RoleBindingsList200Response role_bindings_list(resource_type, limit=limit, cursor=cursor, user_id=user_id)

List role bindings

List role bindings for the authenticated user's account, filtered by
resource type. Results are paginated; use `limit` and `cursor` for
subsequent pages.

The `resource_type` query parameter is **required** and must be one of
`SPACE` or `PROJECT`. All bindings in the account are visible to any
authenticated account member. Use `user_id` to narrow to a specific
user.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.role_binding_resource_type import RoleBindingResourceType
from arize._generated.api_client.models.role_bindings_list200_response import RoleBindingsList200Response
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
    api_instance = arize._generated.api_client.RoleBindingsApi(api_client)
    resource_type = arize._generated.api_client.RoleBindingResourceType() # RoleBindingResourceType | Filter role bindings by resource type. - `SPACE` — Return only space-level bindings. - `PROJECT` — Return only project-level bindings. 
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)
    user_id = 'VXNlcjoxOmxQZzI=' # str | Filter role bindings by user. When provided, only bindings assigned to this user are returned. Must be a valid global user ID.  (optional)

    try:
        # List role bindings
        api_response = api_instance.role_bindings_list(resource_type, limit=limit, cursor=cursor, user_id=user_id)
        print("The response of RoleBindingsApi->role_bindings_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling RoleBindingsApi->role_bindings_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **resource_type** | [**RoleBindingResourceType**](.md)| Filter role bindings by resource type. - &#x60;SPACE&#x60; — Return only space-level bindings. - &#x60;PROJECT&#x60; — Return only project-level bindings.  | 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 
 **user_id** | **str**| Filter role bindings by user. When provided, only bindings assigned to this user are returned. Must be a valid global user ID.  | [optional] 

### Return type

[**RoleBindingsList200Response**](RoleBindingsList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of role binding objects. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **role_bindings_update**
> RoleBinding role_bindings_update(binding_id, role_binding_update)

Update a role binding

Update an existing role binding by changing its assigned role.

**Payload Requirements**
- `role_id` is required and replaces the currently assigned role.
- Only `role_id` is mutable. The binding identity, principal, resource,
  and timestamps stay the same.
- System-managed fields (`id`, `user_id`, `resource_type`,
  `resource_id`, `created_at`, `updated_at`) are not accepted in the
  request body.

**Valid example**
```json
{
  "role_id": "Um9sZToyOmRLMjQ="
}
```

**Invalid example**
```json
{
  "role_id": "Um9sZToyOmRLMjQ=",
  "user_id": "VXNlcjoxOmxQZzI="
}
```
This fails because only `role_id` can be updated on an existing binding.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.role_binding import RoleBinding
from arize._generated.api_client.models.role_binding_update import RoleBindingUpdate
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
    api_instance = arize._generated.api_client.RoleBindingsApi(api_client)
    binding_id = 'Um9sZUJpbmRpbmc6MTphQmNE' # str | The unique role binding identifier (base64)
    role_binding_update = {"role_id":"Um9sZToyOmFCY0Q="} # RoleBindingUpdate | Body containing role binding update parameters.

    try:
        # Update a role binding
        api_response = api_instance.role_bindings_update(binding_id, role_binding_update)
        print("The response of RoleBindingsApi->role_bindings_update:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling RoleBindingsApi->role_bindings_update: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **binding_id** | **str**| The unique role binding identifier (base64) | 
 **role_binding_update** | [**RoleBindingUpdate**](RoleBindingUpdate.md)| Body containing role binding update parameters. | 

### Return type

[**RoleBinding**](RoleBinding.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A role binding object. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

