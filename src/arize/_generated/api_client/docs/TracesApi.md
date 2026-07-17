# arize._generated.api_client.TracesApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**list_traces**](TracesApi.md#list_traces) | **POST** /v2/traces | List traces


# **list_traces**
> ListTracesResponse list_traces(list_traces_request, limit=limit, cursor=cursor)

List traces

Returns a paginated list of traces for a project, each carrying its full
(flat) list of spans plus lightweight roll-up metadata. It accepts the
same `project_id`, `filter`, and time-range parameters as `POST /v2/spans`;
the `filter` uses the identical expression syntax, so there's no separate
filter language to learn.

**Filtering is trace-contains-match**: the syntax matches `/v2/spans`, but
the semantics differ — a `filter` selects traces that contain at least one
matching span (e.g. `status_code = 'ERROR'` or `span_kind = 'LLM'`), not
only traces whose root span matches. The matching span is usually a child,
not the root.

Traces are returned newest-first.

**Behaviors and limitations**
- Traces are anchored on their root span (the span with no parent). A
  trace with no root span in the requested time window is omitted.
- Trace assembly is scoped to the requested time window: spans of a
  boundary-straddling trace that fall outside the range are not included.
- A trace with more than one root span is returned as multiple entries
  sharing the same `trace_id`, distinguished by `root_span_id`.
- Each trace returns at most 1,000 spans. When a trace has more, its
  `spans_truncated` flag is `true`.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_traces_request import ListTracesRequest
from arize._generated.api_client.models.list_traces_response import ListTracesResponse
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
    api_instance = arize._generated.api_client.TracesApi(api_client)
    list_traces_request = {"project_id":"my-project","start_time":"2024-01-01T00:00:00Z","end_time":"2024-01-02T00:00:00Z","filter":"status_code = 'ERROR'"} # ListTracesRequest | Body containing trace query parameters
    limit = 25 # int | Maximum items to return (optional) (default to 25)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List traces
        api_response = api_instance.list_traces(list_traces_request, limit=limit, cursor=cursor)
        print("The response of TracesApi->list_traces:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TracesApi->list_traces: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **list_traces_request** | [**ListTracesRequest**](ListTracesRequest.md)| Body containing trace query parameters | 
 **limit** | **int**| Maximum items to return | [optional] [default to 25]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListTracesResponse**](ListTracesResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of traces |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

