# arize._generated.api_client.RolesApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**roles_create**](RolesApi.md#roles_create) | **POST** /v2/roles | Create a role
[**roles_delete**](RolesApi.md#roles_delete) | **DELETE** /v2/roles/{role_id} | Delete a role
[**roles_get**](RolesApi.md#roles_get) | **GET** /v2/roles/{role_id} | Get a role
[**roles_list**](RolesApi.md#roles_list) | **GET** /v2/roles | List roles
[**roles_update**](RolesApi.md#roles_update) | **PATCH** /v2/roles/{role_id} | Update a role


# **roles_create**
> Role roles_create(role_create)

Create a role

Create a new custom role for the authenticated user's account.

**Payload Requirements**
- `name` is required and must be unique within the account.
- `permissions` is required and must contain at least one valid permission
  identifier (e.g. `PROJECT_READ`, `DATASET_CREATE`).
- System-managed fields (`id`, `created_at`, `updated_at`, `is_predefined`)
  are rejected if provided.

**Valid example**
```json
{
  "name": "Data Scientist",
  "description": "Can read and create datasets and experiments.",
  "permissions": ["PROJECT_READ", "DATASET_READ", "DATASET_CREATE"]
}
```

**Invalid example** (missing required `permissions`)
```json
{
  "name": "Data Scientist"
}
```

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.role import Role
from arize._generated.api_client.models.role_create import RoleCreate
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
    api_instance = arize._generated.api_client.RolesApi(api_client)
    role_create = {"name":"AI Engineer","permissions":["PROJECT_READ","DATASET_READ","DATASET_CREATE"]} # RoleCreate | Body containing role creation parameters.

    try:
        # Create a role
        api_response = api_instance.roles_create(role_create)
        print("The response of RolesApi->roles_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling RolesApi->roles_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **role_create** | [**RoleCreate**](RoleCreate.md)| Body containing role creation parameters. | 

### Return type

[**Role**](Role.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Role successfully created. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **roles_delete**
> roles_delete(role_id)

Delete a role

Delete a custom role by its ID (soft-delete). Predefined roles cannot
be deleted.

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
    api_instance = arize._generated.api_client.RolesApi(api_client)
    role_id = 'Rol001' # str | The unique identifier of the role.

    try:
        # Delete a role
        api_instance.roles_delete(role_id)
    except Exception as e:
        print("Exception when calling RolesApi->roles_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **role_id** | **str**| The unique identifier of the role. | 

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
**204** | Role successfully deleted. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **roles_get**
> Role roles_get(role_id)

Get a role

Get a role by its ID.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.role import Role
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
    api_instance = arize._generated.api_client.RolesApi(api_client)
    role_id = 'Rol001' # str | The unique identifier of the role.

    try:
        # Get a role
        api_response = api_instance.roles_get(role_id)
        print("The response of RolesApi->roles_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling RolesApi->roles_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **role_id** | **str**| The unique identifier of the role. | 

### Return type

[**Role**](Role.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A role object. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **roles_list**
> RolesList200Response roles_list(limit=limit, cursor=cursor, is_predefined=is_predefined)

List roles

List custom and predefined roles for the authenticated user's account.
Results are paginated; use `limit` and `cursor` for subsequent pages.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.roles_list200_response import RolesList200Response
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
    api_instance = arize._generated.api_client.RolesApi(api_client)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)
    is_predefined = True # bool | Filter roles by predefined status. - `true` - Return only system-defined predefined roles. - `false` - Return only custom (account-defined) roles.  When not specified, returns all roles (both predefined and custom).  (optional)

    try:
        # List roles
        api_response = api_instance.roles_list(limit=limit, cursor=cursor, is_predefined=is_predefined)
        print("The response of RolesApi->roles_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling RolesApi->roles_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 
 **is_predefined** | **bool**| Filter roles by predefined status. - &#x60;true&#x60; - Return only system-defined predefined roles. - &#x60;false&#x60; - Return only custom (account-defined) roles.  When not specified, returns all roles (both predefined and custom).  | [optional] 

### Return type

[**RolesList200Response**](RolesList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of role objects. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **roles_update**
> Role roles_update(role_id, role_update)

Update a role

Update a custom role by its ID. At least one field must be provided.
Predefined roles cannot be updated.

**Payload Requirements**
- At least one of `name`, `description`, or `permissions` must be provided.
- When `permissions` is provided, the existing permissions are fully replaced with the new set.
- `name`, if provided, must be unique within the account.
- System-managed fields (`id`, `created_at`, `updated_at`, `is_predefined`) cannot be modified.

**Valid example**
```json
{
  "name": "Senior Data Scientist",
  "permissions": ["PROJECT_READ", "DATASET_READ", "DATASET_CREATE", "DATASET_DELETE"]
}
```

**Invalid example** (no fields provided)
```json
{}
```

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.role import Role
from arize._generated.api_client.models.role_update import RoleUpdate
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
    api_instance = arize._generated.api_client.RolesApi(api_client)
    role_id = 'Rol001' # str | The unique identifier of the role.
    role_update = {"name":"Senior AI Engineer"} # RoleUpdate | Body containing role update parameters. At least one field must be provided.

    try:
        # Update a role
        api_response = api_instance.roles_update(role_id, role_update)
        print("The response of RolesApi->roles_update:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling RolesApi->roles_update: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **role_id** | **str**| The unique identifier of the role. | 
 **role_update** | [**RoleUpdate**](RoleUpdate.md)| Body containing role update parameters. At least one field must be provided. | 

### Return type

[**Role**](Role.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A role object. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

