# arize._generated.api_client.SpansApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**spans_delete**](SpansApi.md#spans_delete) | **DELETE** /v2/spans | Delete spans
[**spans_list**](SpansApi.md#spans_list) | **POST** /v2/spans | List spans


# **spans_delete**
> SpansDelete200Response spans_delete(spans_delete_request)

Delete spans

Permanently deletes spans by their span IDs. This operation is irreversible.

Accepts between 1 and 1000 span IDs per request. Only spans from the
last 31 days are considered; older spans are not affected.

A `204 No Content` response indicates all extant IDs provided
within the last 31 days were deleted.

A `200 OK` response indicates one or more intervals could not be fully processed
within the retry budget. Retry the original request for a correct result.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.spans_delete200_response import SpansDelete200Response
from arize._generated.api_client.models.spans_delete_request import SpansDeleteRequest
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
    api_instance = arize._generated.api_client.SpansApi(api_client)
    spans_delete_request = {"project_id":"UHJvamVjdDox","span_ids":["a1b2c3d4e5f6a7b8","f8e7d6c5b4a39281"]} # SpansDeleteRequest | Body containing span IDs to delete

    try:
        # Delete spans
        api_response = api_instance.spans_delete(spans_delete_request)
        print("The response of SpansApi->spans_delete:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SpansApi->spans_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **spans_delete_request** | [**SpansDeleteRequest**](SpansDeleteRequest.md)| Body containing span IDs to delete | 

### Return type

[**SpansDelete200Response**](SpansDelete200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Some span IDs could not be confirmed deleted within the allotted retries. Retry the original request for a completed deletion result.  |  -  |
**204** | Spans successfully deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |
**500** | Fatal mid-request error. Body carries any IDs already confirmed deleted. |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **spans_list**
> SpansList200Response spans_list(spans_list_request, limit=limit, cursor=cursor)

List spans

Returns a paginated list of spans.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.spans_list200_response import SpansList200Response
from arize._generated.api_client.models.spans_list_request import SpansListRequest
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
    api_instance = arize._generated.api_client.SpansApi(api_client)
    spans_list_request = {"project_id":"my-project","start_time":"2024-01-01T00:00:00Z","end_time":"2024-01-02T00:00:00Z","filter":"status_code = 'ERROR'"} # SpansListRequest | Body containing span query parameters
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List spans
        api_response = api_instance.spans_list(spans_list_request, limit=limit, cursor=cursor)
        print("The response of SpansApi->spans_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SpansApi->spans_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **spans_list_request** | [**SpansListRequest**](SpansListRequest.md)| Body containing span query parameters | 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**SpansList200Response**](SpansList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of spans |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

