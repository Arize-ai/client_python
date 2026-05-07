# arize._generated.api_client.SpansApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**spans_annotate**](SpansApi.md#spans_annotate) | **POST** /v2/spans/annotate | Annotate a batch of project spans
[**spans_delete**](SpansApi.md#spans_delete) | **DELETE** /v2/spans | Delete spans
[**spans_list**](SpansApi.md#spans_list) | **POST** /v2/spans | List spans


# **spans_annotate**
> spans_annotate(annotate_spans_request_body)

Annotate a batch of project spans

Write human annotations to a batch of spans in a project.

**Idempotency**: Writes use upsert semantics — submitting the same annotation
config name for the same span overwrites the previous value. Retrying on
network failure will not create duplicates.

**202 Accepted**: The request was validated and the writes were submitted
to the database layer. Changes may not be immediately visible in queries.

**Partial failure**: This endpoint writes records in day-bucket batches.
A non-2xx response means the request failed partway through — some records
may already be saved and some may not. It is safe to retry the full
request; re-submitting a record that was already saved will overwrite it
with the same value (no duplicates).

**Payload Requirements**
- `project_id` is required and must identify a project the caller has span annotation access to.
- `annotations` is a list of per-span annotation inputs. Each entry identifies
  one span by its `record_id` and provides one or more annotation values.
- `start_time` / `end_time` constrain the Druid time range for span lookup.
  If omitted, `start_time` defaults to 7 days ago and `end_time` to now.
  The window may not exceed 31 days and `end_time` may not be in the future.
  If ANY span ID cannot be located within the given range, the entire
  request is rejected with 404 and no annotations are written (all-or-nothing
  pre-validation).
- Annotation names must match existing annotation configs in the project's space.
- Up to 1000 span records may be annotated per request.

**Valid example**
```json
{
  "project_id": "proj_abc123",
  "annotations": [
    {"record_id": "span_abc", "values": [{"name": "relevance", "label": "good", "score": 1.0}]}
  ]
}
```

**Invalid example** (annotation name not found in space)
```json
{
  "project_id": "proj_abc123",
  "annotations": [
    {"record_id": "span_abc", "values": [{"name": "nonexistent_config"}]}
  ]
}
```

**Invalid example** (time window exceeds 31 days)
```json
{
  "project_id": "proj_abc123",
  "start_time": "2025-01-01T00:00:00Z",
  "end_time": "2025-03-01T00:00:00Z",
  "annotations": [
    {"record_id": "span_abc", "values": [{"name": "relevance", "label": "good"}]}
  ]
}
```

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotate_spans_request_body import AnnotateSpansRequestBody
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
    annotate_spans_request_body = {"project_id":"proj_abc123","start_time":"2024-01-01T00:00:00Z","end_time":"2024-01-08T00:00:00Z","annotations":[{"record_id":"span_abc","values":[{"name":"relevance","label":"good","score":1.5}]}]} # AnnotateSpansRequestBody | Body containing span annotation batch

    try:
        # Annotate a batch of project spans
        api_instance.spans_annotate(annotate_spans_request_body)
    except Exception as e:
        print("Exception when calling SpansApi->spans_annotate: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotate_spans_request_body** | [**AnnotateSpansRequestBody**](AnnotateSpansRequestBody.md)| Body containing span annotation batch | 

### Return type

void (empty response body)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**202** | Annotations submitted successfully. The request was validated and the writes were submitted to the database layer. Changes may not be immediately visible in queries. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **spans_delete**
> SpansDelete200Response spans_delete(spans_delete_request)

Delete spans

Permanently deletes spans by their span IDs. This operation is irreversible.

Accepts between 1 and 5000 span IDs per request. Only spans within the
supported lookback window are considered; older spans are not affected.

A `204 No Content` response indicates all extant IDs provided
within the supported lookback window were deleted.

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

