# arize._generated.api_client.SpacesApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**add_space_user**](SpacesApi.md#add_space_user) | **POST** /v2/spaces/{space_id}/users | Add a user to a space
[**create_space**](SpacesApi.md#create_space) | **POST** /v2/spaces | Create a space
[**delete_space**](SpacesApi.md#delete_space) | **DELETE** /v2/spaces/{space_id} | Delete a space
[**get_space**](SpacesApi.md#get_space) | **GET** /v2/spaces/{space_id} | Get a space
[**list_spaces**](SpacesApi.md#list_spaces) | **GET** /v2/spaces | List spaces
[**remove_space_user**](SpacesApi.md#remove_space_user) | **DELETE** /v2/spaces/{space_id}/users/{user_id} | Remove a user from a space
[**update_space**](SpacesApi.md#update_space) | **PATCH** /v2/spaces/{space_id} | Update a space


# **add_space_user**
> SpaceMembership add_space_user(space_id, add_space_user_request)

Add a user to a space

Add a single existing account user to a space with a specified role.

**Payload Requirements**
- `user_id` is required and must be a valid user identifier (base64).
- `role` is required and must be a role assignment object with a `type` discriminator:
  - `{ "type": "PREDEFINED", "name": "ADMIN" }` — one of the predefined roles: `ADMIN`, `MEMBER`, `READ_ONLY`, `ANNOTATOR`.
  - `{ "type": "CUSTOM", "id": "<role_id>" }` — a custom RBAC role, using its unique identifier.
- If the user is already a member, their role is updated to the specified value (upsert).
- The user must already be a member of the space's parent organization; auto-enrollment is not performed (400 if not a member).

**Role constraints**
- Users with an `annotator` account role can only be assigned the `annotator` predefined space role.
- Users with a non-annotator account role cannot be assigned the `annotator` predefined space role.

**Authorization**
Requires space admin role when using a `PREDEFINED` role, or `ROLE_BINDING_CREATE`
permission (RBAC) when using a `CUSTOM` role.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.add_space_user_request import AddSpaceUserRequest
from arize._generated.api_client.models.space_membership import SpaceMembership
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
    space_id = 'U3BhY2U6MTIzNDU=' # str | The unique space identifier (base64)
    add_space_user_request = {"user_id":"VXNlcjoxMjM0NQ==","role":{"type":"PREDEFINED","name":"MEMBER"}} # AddSpaceUserRequest | Body containing the user to add to the space

    try:
        # Add a user to a space
        api_response = api_instance.add_space_user(space_id, add_space_user_request)
        print("The response of SpacesApi->add_space_user:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SpacesApi->add_space_user: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| The unique space identifier (base64) | 
 **add_space_user_request** | [**AddSpaceUserRequest**](AddSpaceUserRequest.md)| Body containing the user to add to the space | 

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
**200** | User successfully added to the space |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **create_space**
> Space create_space(create_space_request)

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
from arize._generated.api_client.models.create_space_request import CreateSpaceRequest
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
    create_space_request = {"name":"LLM Evaluation","organization_id":"org_12345","description":"Space for evaluating LLM performance"} # CreateSpaceRequest | Body containing space creation parameters

    try:
        # Create a space
        api_response = api_instance.create_space(create_space_request)
        print("The response of SpacesApi->create_space:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SpacesApi->create_space: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_space_request** | [**CreateSpaceRequest**](CreateSpaceRequest.md)| Body containing space creation parameters | 

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
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_space**
> delete_space(space_id)

Delete a space

Delete a space by its ID. This deletes the space and all resources
that belong to it (models, monitors, dashboards, datasets, custom metrics, etc).
This operation is irreversible.

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
    api_instance = arize._generated.api_client.SpacesApi(api_client)
    space_id = 'U3BhY2U6MTIzNDU=' # str | The unique space identifier (base64)

    try:
        # Delete a space
        api_instance.delete_space(space_id)
    except Exception as e:
        print("Exception when calling SpacesApi->delete_space: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| The unique space identifier (base64) | 

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

# **get_space**
> Space get_space(space_id)

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
    space_id = 'U3BhY2U6MTIzNDU=' # str | The unique space identifier (base64)

    try:
        # Get a space
        api_response = api_instance.get_space(space_id)
        print("The response of SpacesApi->get_space:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SpacesApi->get_space: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| The unique space identifier (base64) | 

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

# **list_spaces**
> ListSpacesResponse list_spaces(org_id=org_id, name=name, limit=limit, cursor=cursor)

List spaces

List spaces the user has access to.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_spaces_response import ListSpacesResponse
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
    org_id = 'T3JnYW5pemF0aW9uOjEyMzQ1' # str | The unique organization identifier (base64). When provided, only spaces belonging to this organization are returned. (optional)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List spaces
        api_response = api_instance.list_spaces(org_id=org_id, name=name, limit=limit, cursor=cursor)
        print("The response of SpacesApi->list_spaces:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SpacesApi->list_spaces: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **org_id** | **str**| The unique organization identifier (base64). When provided, only spaces belonging to this organization are returned. | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. If omitted, no name filtering is applied and all resources are returned.  | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListSpacesResponse**](ListSpacesResponse.md)

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

# **remove_space_user**
> remove_space_user(space_id, user_id)

Remove a user from a space

Remove a user from a space. This removes both the legacy `SpaceMembers` row
and any RBAC role bindings for the user on this space.

Returns 404 if the user is not a member of the space.

**Authorization**
Requires space admin role (legacy auth) or `ROLE_BINDING_DELETE` permission (RBAC).

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
    api_instance = arize._generated.api_client.SpacesApi(api_client)
    space_id = 'U3BhY2U6MTIzNDU=' # str | The unique space identifier (base64)
    user_id = 'VXNlcjoxMjM0NQ==' # str | The unique user identifier (base64)

    try:
        # Remove a user from a space
        api_instance.remove_space_user(space_id, user_id)
    except Exception as e:
        print("Exception when calling SpacesApi->remove_space_user: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| The unique space identifier (base64) | 
 **user_id** | **str**| The unique user identifier (base64) | 

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

# **update_space**
> Space update_space(space_id, update_space_request)

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
from arize._generated.api_client.models.update_space_request import UpdateSpaceRequest
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
    space_id = 'U3BhY2U6MTIzNDU=' # str | The unique space identifier (base64)
    update_space_request = {"name":"Updated Space Name","description":"Updated space description"} # UpdateSpaceRequest | Body containing space update parameters. At least one field must be provided.

    try:
        # Update a space
        api_response = api_instance.update_space(space_id, update_space_request)
        print("The response of SpacesApi->update_space:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SpacesApi->update_space: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| The unique space identifier (base64) | 
 **update_space_request** | [**UpdateSpaceRequest**](UpdateSpaceRequest.md)| Body containing space update parameters. At least one field must be provided. | 

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
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

