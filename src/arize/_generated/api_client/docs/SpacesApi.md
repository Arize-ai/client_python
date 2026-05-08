# arize._generated.api_client.SpacesApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**spaces_add_user**](SpacesApi.md#spaces_add_user) | **POST** /v2/spaces/{space_id}/users | Add a user to a space
[**spaces_create**](SpacesApi.md#spaces_create) | **POST** /v2/spaces | Create a space
[**spaces_delete**](SpacesApi.md#spaces_delete) | **DELETE** /v2/spaces/{space_id} | Delete a space
[**spaces_get**](SpacesApi.md#spaces_get) | **GET** /v2/spaces/{space_id} | Get a space
[**spaces_list**](SpacesApi.md#spaces_list) | **GET** /v2/spaces | List spaces
[**spaces_remove_user**](SpacesApi.md#spaces_remove_user) | **DELETE** /v2/spaces/{space_id}/users/{user_id} | Remove a user from a space
[**spaces_update**](SpacesApi.md#spaces_update) | **PATCH** /v2/spaces/{space_id} | Update a space


# **spaces_add_user**
> SpaceMembership spaces_add_user(space_id, space_membership_input)

Add a user to a space

Add a single existing account user to a space with a specified role.

**Payload Requirements**
- `user_id` is required and must be a valid User global ID.
- `role` is required and must be a role assignment object with a `type` discriminator:
  - `{ "type": "builtin", "name": "admin" }` — one of the predefined roles: `admin`, `member`, `read-only`, `annotator`.
  - `{ "type": "custom", "id": "<role_id>" }` — a custom RBAC role identified by its global ID.
- If the user is already a member, their role is updated to the specified value (upsert).
- The user must already be a member of the space's parent organization; auto-enrollment is not performed (400 if not a member).

**Role constraints**
- Users with an `annotator` account role can only be assigned the `annotator` builtin space role.
- Users with a non-annotator account role cannot be assigned the `annotator` builtin space role.

**Authorization**
Requires space admin role when using a `builtin` role, or `ROLE_BINDING_CREATE`
permission (RBAC) when using a `custom` role.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.space_membership import SpaceMembership
from arize._generated.api_client.models.space_membership_input import SpaceMembershipInput
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
    api_instance = arize._generated.api_client.SpacesApi(api_client)
    space_id = 'spc_12345' # str | The unique identifier of the space
    space_membership_input = {"user_id":"VXNlcjoxMjM0NQ==","role":{"type":"builtin","name":"member"}} # SpaceMembershipInput | Body containing the user to add to the space

    try:
        # Add a user to a space
        api_response = api_instance.spaces_add_user(space_id, space_membership_input)
        print("The response of SpacesApi->spaces_add_user:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SpacesApi->spaces_add_user: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| The unique identifier of the space | 
 **space_membership_input** | [**SpaceMembershipInput**](SpaceMembershipInput.md)| Body containing the user to add to the space | 

### Return type

[**SpaceMembership**](SpaceMembership.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | User successfully added to the space |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **spaces_create**
> Space spaces_create(spaces_create_request)

Create a space

Create a new space within an organization.

**Payload Requirements**
- `name` and `organization_id` are required.
- The space name must be unique within the organization.
- `description` is optional and defaults to an empty string if omitted.
- System-managed fields (`id`, `created_at`) are generated
  automatically and rejected if provided.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.space import Space
from arize._generated.api_client.models.spaces_create_request import SpacesCreateRequest
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
    api_instance = arize._generated.api_client.SpacesApi(api_client)
    spaces_create_request = {"name":"LLM Evaluation","organization_id":"org_12345","description":"Space for evaluating LLM performance"} # SpacesCreateRequest | Body containing space creation parameters

    try:
        # Create a space
        api_response = api_instance.spaces_create(spaces_create_request)
        print("The response of SpacesApi->spaces_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SpacesApi->spaces_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **spaces_create_request** | [**SpacesCreateRequest**](SpacesCreateRequest.md)| Body containing space creation parameters | 

### Return type

[**Space**](Space.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | A space object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **spaces_delete**
> spaces_delete(space_id)

Delete a space

Delete a space by its ID. This deletes the space and all resources
that belong to it (models, monitors, dashboards, datasets, custom metrics, etc).
This operation is irreversible.

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
    api_instance = arize._generated.api_client.SpacesApi(api_client)
    space_id = 'spc_12345' # str | The unique identifier of the space

    try:
        # Delete a space
        api_instance.spaces_delete(space_id)
    except Exception as e:
        print("Exception when calling SpacesApi->spaces_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| The unique identifier of the space | 

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
**204** | Space successfully deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **spaces_get**
> Space spaces_get(space_id)

Get a space

Get a specific space by its ID.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.space import Space
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
    api_instance = arize._generated.api_client.SpacesApi(api_client)
    space_id = 'spc_12345' # str | The unique identifier of the space

    try:
        # Get a space
        api_response = api_instance.spaces_get(space_id)
        print("The response of SpacesApi->spaces_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SpacesApi->spaces_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| The unique identifier of the space | 

### Return type

[**Space**](Space.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A space object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **spaces_list**
> SpacesList200Response spaces_list(org_id=org_id, name=name, limit=limit, cursor=cursor)

List spaces

List spaces the user has access to.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.spaces_list200_response import SpacesList200Response
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
    api_instance = arize._generated.api_client.SpacesApi(api_client)
    org_id = 'org_id_example' # str | The unique identifier of an organization. When provided, only spaces belonging to this organization are returned. (optional)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List spaces
        api_response = api_instance.spaces_list(org_id=org_id, name=name, limit=limit, cursor=cursor)
        print("The response of SpacesApi->spaces_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SpacesApi->spaces_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **org_id** | **str**| The unique identifier of an organization. When provided, only spaces belonging to this organization are returned. | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. If omitted, no name filtering is applied and all resources are returned.  | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**SpacesList200Response**](SpacesList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of space objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **spaces_remove_user**
> spaces_remove_user(space_id, user_id)

Remove a user from a space

Remove a user from a space. This removes both the legacy `SpaceMembers` row
and any RBAC role bindings for the user on this space.

Returns 404 if the user is not a member of the space.

**Authorization**
Requires space admin role (legacy auth) or `ROLE_BINDING_DELETE` permission (RBAC).

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
    api_instance = arize._generated.api_client.SpacesApi(api_client)
    space_id = 'spc_12345' # str | The unique identifier of the space
    user_id = 'VXNlcjoxMjM0NQ==' # str | The unique identifier of the user

    try:
        # Remove a user from a space
        api_instance.spaces_remove_user(space_id, user_id)
    except Exception as e:
        print("Exception when calling SpacesApi->spaces_remove_user: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| The unique identifier of the space | 
 **user_id** | **str**| The unique identifier of the user | 

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
**204** | User successfully removed from the space |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **spaces_update**
> Space spaces_update(space_id, spaces_update_request)

Update a space

Update a space's metadata by its ID. Currently supports updating the
name and description. At least one field must be provided.

**Payload Requirements**
- At least one of `name` or `description` must be provided.
- If `name` is provided, it must be unique within the organization.
- System-managed fields (`id`, `created_at`) cannot be modified.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.space import Space
from arize._generated.api_client.models.spaces_update_request import SpacesUpdateRequest
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
    api_instance = arize._generated.api_client.SpacesApi(api_client)
    space_id = 'spc_12345' # str | The unique identifier of the space
    spaces_update_request = {"name":"Updated Space Name","description":"Updated space description"} # SpacesUpdateRequest | Body containing space update parameters. At least one field must be provided.

    try:
        # Update a space
        api_response = api_instance.spaces_update(space_id, spaces_update_request)
        print("The response of SpacesApi->spaces_update:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SpacesApi->spaces_update: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| The unique identifier of the space | 
 **spaces_update_request** | [**SpacesUpdateRequest**](SpacesUpdateRequest.md)| Body containing space update parameters. At least one field must be provided. | 

### Return type

[**Space**](Space.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A space object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

