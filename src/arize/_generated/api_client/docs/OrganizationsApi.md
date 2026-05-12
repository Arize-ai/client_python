# arize._generated.api_client.OrganizationsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**organizations_add_user**](OrganizationsApi.md#organizations_add_user) | **POST** /v2/organizations/{org_id}/users | Add a user to an organization
[**organizations_create**](OrganizationsApi.md#organizations_create) | **POST** /v2/organizations | Create an organization
[**organizations_delete**](OrganizationsApi.md#organizations_delete) | **DELETE** /v2/organizations/{org_id} | Delete an organization
[**organizations_get**](OrganizationsApi.md#organizations_get) | **GET** /v2/organizations/{org_id} | Get an organization
[**organizations_list**](OrganizationsApi.md#organizations_list) | **GET** /v2/organizations | List organizations
[**organizations_remove_user**](OrganizationsApi.md#organizations_remove_user) | **DELETE** /v2/organizations/{org_id}/users/{user_id} | Remove a user from an organization
[**organizations_update**](OrganizationsApi.md#organizations_update) | **PATCH** /v2/organizations/{org_id} | Update an organization


# **organizations_add_user**
> OrganizationMembership organizations_add_user(org_id, organization_membership_input)

Add a user to an organization

Add a single existing account user to an organization with a specified role.

**Payload Requirements**
- `user_id` and `role` are both required.
- If the user is already a member, their role is updated to the specified value (upsert).

**Role constraints**
- Users with an `annotator` account role can only be assigned the `annotator` organization role.
- Users with a non-annotator account role cannot be assigned the `annotator` organization role.

Requires organization admin.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.organization_membership import OrganizationMembership
from arize._generated.api_client.models.organization_membership_input import OrganizationMembershipInput
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
    api_instance = arize._generated.api_client.OrganizationsApi(api_client)
    org_id = 'org_12345' # str | The unique identifier of the organization
    organization_membership_input = {"user_id":"VXNlcjoxMjM0NQ==","role":{"type":"predefined","name":"member"}} # OrganizationMembershipInput | Body containing the user to add to the organization

    try:
        # Add a user to an organization
        api_response = api_instance.organizations_add_user(org_id, organization_membership_input)
        print("The response of OrganizationsApi->organizations_add_user:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling OrganizationsApi->organizations_add_user: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **org_id** | **str**| The unique identifier of the organization | 
 **organization_membership_input** | [**OrganizationMembershipInput**](OrganizationMembershipInput.md)| Body containing the user to add to the organization | 

### Return type

[**OrganizationMembership**](OrganizationMembership.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | User successfully added to the organization |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **organizations_create**
> Organization organizations_create(organizations_create_request)

Create an organization

Create a new organization within the account.

**Payload Requirements**
- `name` is required and must be unique within the account.
- `description` is optional and defaults to an empty string if omitted.
- System-managed fields (`id`, `created_at`) are generated automatically and rejected if provided.

**Example valid request:**
```json
{ "name": "Agent Engineering Team", "description": "Organization for the Agent Engineering team" }
```

**Example invalid request (duplicate name):**
```json
{ "name": "Agent Engineering Team" }
```
Fails with 409 Conflict if an organization with that name already exists in the account.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.organization import Organization
from arize._generated.api_client.models.organizations_create_request import OrganizationsCreateRequest
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
    api_instance = arize._generated.api_client.OrganizationsApi(api_client)
    organizations_create_request = {"name":"Agent Engineering Team","description":"Organization for the Agent Engineering team"} # OrganizationsCreateRequest | Body containing organization creation parameters

    try:
        # Create an organization
        api_response = api_instance.organizations_create(organizations_create_request)
        print("The response of OrganizationsApi->organizations_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling OrganizationsApi->organizations_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **organizations_create_request** | [**OrganizationsCreateRequest**](OrganizationsCreateRequest.md)| Body containing organization creation parameters | 

### Return type

[**Organization**](Organization.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | An organization object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **organizations_delete**
> organizations_delete(org_id)

Delete an organization

Delete an organization by its ID. This deletes the organization and all
resources that belong to it, including all spaces and their contents
(projects, experiments, evaluators, models, monitors, dashboards, datasets,
annotation configs, annotation queues, custom metrics, etc.) as well
as organization-level resources such as integrations, cost configurations,
SAML identity providers, and API keys.

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
    api_instance = arize._generated.api_client.OrganizationsApi(api_client)
    org_id = 'org_12345' # str | The unique identifier of the organization

    try:
        # Delete an organization
        api_instance.organizations_delete(org_id)
    except Exception as e:
        print("Exception when calling OrganizationsApi->organizations_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **org_id** | **str**| The unique identifier of the organization | 

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
**204** | Organization successfully deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **organizations_get**
> Organization organizations_get(org_id)

Get an organization

Get a specific organization by its ID.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.organization import Organization
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
    api_instance = arize._generated.api_client.OrganizationsApi(api_client)
    org_id = 'org_12345' # str | The unique identifier of the organization

    try:
        # Get an organization
        api_response = api_instance.organizations_get(org_id)
        print("The response of OrganizationsApi->organizations_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling OrganizationsApi->organizations_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **org_id** | **str**| The unique identifier of the organization | 

### Return type

[**Organization**](Organization.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An organization object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **organizations_list**
> OrganizationsList200Response organizations_list(name=name, limit=limit, cursor=cursor)

List organizations

List organizations the user has access to.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.organizations_list200_response import OrganizationsList200Response
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
    api_instance = arize._generated.api_client.OrganizationsApi(api_client)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List organizations
        api_response = api_instance.organizations_list(name=name, limit=limit, cursor=cursor)
        print("The response of OrganizationsApi->organizations_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling OrganizationsApi->organizations_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. If omitted, no name filtering is applied and all resources are returned.  | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**OrganizationsList200Response**](OrganizationsList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of organization objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **organizations_remove_user**
> organizations_remove_user(org_id, user_id)

Remove a user from an organization

Remove a user from the organization and all its child spaces (membership cascade).

Requires organization admin.

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
    api_instance = arize._generated.api_client.OrganizationsApi(api_client)
    org_id = 'org_12345' # str | The unique identifier of the organization
    user_id = 'VXNlcjoxMjM0NQ==' # str | The unique identifier of the user

    try:
        # Remove a user from an organization
        api_instance.organizations_remove_user(org_id, user_id)
    except Exception as e:
        print("Exception when calling OrganizationsApi->organizations_remove_user: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **org_id** | **str**| The unique identifier of the organization | 
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
**204** | User successfully removed from the organization |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **organizations_update**
> Organization organizations_update(org_id, organizations_update_request)

Update an organization

Update an organization's metadata by its ID. At least one field must be provided.

**Payload Requirements**
- At least one of `name` or `description` must be provided.
- If `name` is provided, it must be unique within the account.
- System-managed fields (`id`, `created_at`) cannot be modified.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.organization import Organization
from arize._generated.api_client.models.organizations_update_request import OrganizationsUpdateRequest
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
    api_instance = arize._generated.api_client.OrganizationsApi(api_client)
    org_id = 'org_12345' # str | The unique identifier of the organization
    organizations_update_request = {"name":"Updated Org Name","description":"Updated organization description"} # OrganizationsUpdateRequest | Body containing organization update parameters. At least one field must be provided.

    try:
        # Update an organization
        api_response = api_instance.organizations_update(org_id, organizations_update_request)
        print("The response of OrganizationsApi->organizations_update:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling OrganizationsApi->organizations_update: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **org_id** | **str**| The unique identifier of the organization | 
 **organizations_update_request** | [**OrganizationsUpdateRequest**](OrganizationsUpdateRequest.md)| Body containing organization update parameters. At least one field must be provided. | 

### Return type

[**Organization**](Organization.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An organization object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

