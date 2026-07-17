# arize._generated.api_client.ResourceRestrictionsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_resource_restriction**](ResourceRestrictionsApi.md#create_resource_restriction) | **POST** /v2/resource-restrictions | Restrict a resource
[**delete_resource_restriction**](ResourceRestrictionsApi.md#delete_resource_restriction) | **DELETE** /v2/resource-restrictions/{resource_id} | Unrestrict a resource
[**list_resource_restrictions**](ResourceRestrictionsApi.md#list_resource_restrictions) | **GET** /v2/resource-restrictions | List resource restrictions the caller is permitted to manage.


# **create_resource_restriction**
> ResourceRestriction create_resource_restriction(create_resource_restriction_request)

Restrict a resource

Mark a resource as restricted. Only space admins or users with the RESOURCE_RESTRICT
permission can perform this action. Idempotent.

**Payload Requirements**
- `resource_id`: The ID for the resource. 
  Only `project` resources are currently supported. Other resource types are not currently supported and will return 400.

**Valid example**
```json
{ "resource_id": "TW9kZWw6MTIxOmFCY0Q=" }
```

**Invalid example**
```json
{ "resource_id": "Not a project ID" }
```
Returns 400 — only Project / Model IDs are accepted 

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_resource_restriction_request import CreateResourceRestrictionRequest
from arize._generated.api_client.models.resource_restriction import ResourceRestriction
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
    api_instance = arize._generated.api_client.ResourceRestrictionsApi(api_client)
    create_resource_restriction_request = {"resource_id":"TW9kZWw6MTIxOmFCY0Q="} # CreateResourceRestrictionRequest | Body containing resource restriction creation parameters.

    try:
        # Restrict a resource
        api_response = api_instance.create_resource_restriction(create_resource_restriction_request)
        print("The response of ResourceRestrictionsApi->create_resource_restriction:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ResourceRestrictionsApi->create_resource_restriction: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_resource_restriction_request** | [**CreateResourceRestrictionRequest**](CreateResourceRestrictionRequest.md)| Body containing resource restriction creation parameters. | 

### Return type

[**ResourceRestriction**](ResourceRestriction.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A resource restriction record |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_resource_restriction**
> delete_resource_restriction(resource_id)

Unrestrict a resource

Remove restriction from a resource. Removing a restriction from a resource means that roles bound at other levels of the hierarchy (space, org, account) can grant access to the resource. Returns 404 if the resource is not restricted.

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
    api_instance = arize._generated.api_client.ResourceRestrictionsApi(api_client)
    resource_id = 'resource_id_example' # str | The unique resource identifier (base64)

    try:
        # Unrestrict a resource
        api_instance.delete_resource_restriction(resource_id)
    except Exception as e:
        print("Exception when calling ResourceRestrictionsApi->delete_resource_restriction: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **resource_id** | **str**| The unique resource identifier (base64) | 

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
**204** | Resource restriction deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_resource_restrictions**
> ListResourceRestrictionsResponse list_resource_restrictions(resource_type=resource_type, limit=limit, cursor=cursor)

List resource restrictions the caller is permitted to manage.

List active resource restrictions the authenticated user is permitted to manage.
A restriction is returned only if the caller can manage it — i.e. an account/org admin
(via admin escalation), a holder of the `PROJECT_RESTRICT` permission in the project's
space, or a holder of `PROJECT_RESTRICT` granted directly on the project.

Results are paginated; use `limit` and `cursor` for subsequent pages. Because entries
are authorization-filtered after a page is read, a page may contain fewer items than
`limit` (or be empty) while `has_more` is still `true`. Clients MUST keep paging until
`has_more` is `false` — do not stop on an empty page.

Use the optional `resource_type` query param to filter to a single resource type.
When omitted, `PROJECT` restrictions are returned (currently the only supported type).

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_resource_restrictions_response import ListResourceRestrictionsResponse
from arize._generated.api_client.models.resource_restriction_type import ResourceRestrictionType
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
    api_instance = arize._generated.api_client.ResourceRestrictionsApi(api_client)
    resource_type = arize._generated.api_client.ResourceRestrictionType() # ResourceRestrictionType | Filter restrictions to a single resource type. - `PROJECT` — Return only restricted projects.  When not specified, restrictions of all supported resource types are returned (currently only `PROJECT`).  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List resource restrictions the caller is permitted to manage.
        api_response = api_instance.list_resource_restrictions(resource_type=resource_type, limit=limit, cursor=cursor)
        print("The response of ResourceRestrictionsApi->list_resource_restrictions:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ResourceRestrictionsApi->list_resource_restrictions: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **resource_type** | [**ResourceRestrictionType**](.md)| Filter restrictions to a single resource type. - &#x60;PROJECT&#x60; — Return only restricted projects.  When not specified, restrictions of all supported resource types are returned (currently only &#x60;PROJECT&#x60;).  | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListResourceRestrictionsResponse**](ListResourceRestrictionsResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A list of resource restriction records. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

