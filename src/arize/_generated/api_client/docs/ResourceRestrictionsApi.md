# arize._generated.api_client.ResourceRestrictionsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**resource_restrictions_create**](ResourceRestrictionsApi.md#resource_restrictions_create) | **POST** /v2/resource-restrictions | Restrict a resource
[**resource_restrictions_delete**](ResourceRestrictionsApi.md#resource_restrictions_delete) | **DELETE** /v2/resource-restrictions/{resource_id} | Unrestrict a resource


# **resource_restrictions_create**
> ResourceRestrictionsCreate200Response resource_restrictions_create(resource_restrictions_create_request)

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

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.resource_restrictions_create200_response import ResourceRestrictionsCreate200Response
from arize._generated.api_client.models.resource_restrictions_create_request import ResourceRestrictionsCreateRequest
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
    resource_restrictions_create_request = {"resource_id":"TW9kZWw6MTIxOmFCY0Q="} # ResourceRestrictionsCreateRequest | 

    try:
        # Restrict a resource
        api_response = api_instance.resource_restrictions_create(resource_restrictions_create_request)
        print("The response of ResourceRestrictionsApi->resource_restrictions_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ResourceRestrictionsApi->resource_restrictions_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **resource_restrictions_create_request** | [**ResourceRestrictionsCreateRequest**](ResourceRestrictionsCreateRequest.md)|  | 

### Return type

[**ResourceRestrictionsCreate200Response**](ResourceRestrictionsCreate200Response.md)

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
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **resource_restrictions_delete**
> resource_restrictions_delete(resource_id)

Unrestrict a resource

Remove restriction from a resource. Removing a restriction from a resource means that roles bound at other levels of the hierarchy (space, org, account) can grant access to the resource. Returns 404 if the resource is not restricted.

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
    api_instance = arize._generated.api_client.ResourceRestrictionsApi(api_client)
    resource_id = 'resource_id_example' # str | The unique identifier of the resource

    try:
        # Unrestrict a resource
        api_instance.resource_restrictions_delete(resource_id)
    except Exception as e:
        print("Exception when calling ResourceRestrictionsApi->resource_restrictions_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **resource_id** | **str**| The unique identifier of the resource | 

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

