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

Trace entries are ordered by root span `start_time` from newest to oldest.
Root trace and span identifiers give entries with the same start time a
stable order. Start and end time bounds are inclusive.

**Behaviors and limitations**
- Traces are anchored on their root span (the span with no parent). A
  trace with no root span in the requested time window is omitted.
- Trace assembly is scoped to the requested time window: spans of a
  boundary-straddling trace that fall outside the range are not included.
- A trace with more than one root span is returned as multiple entries
  sharing the same `trace_id`, distinguished by `root_span_id`.
- Each trace returns at most 1,000 spans. Traces share a fetch allowance
  per page. When that allowance is exhausted, traces can be incomplete
  even when `spans_truncated` is `false`.

Use the returned cursor with the same project, filter, and time window.
You can change the page limit. If the server rejects a cursor after an
endpoint update, restart the page walk without it. Cursor pagination keeps
one time window fixed, but it is not a snapshot of changing data.

**Traces that arrive long after they started**

`start_time` is the time your application recorded for the span. Arize
also stores the time it received the span. This endpoint searches
received-time storage for a few hours on either side of the `start_time`
range you ask for, which is how the Arize UI reads the same data.

A trace that reached Arize much later than it started can therefore fall
outside that search. Backfilled or replayed traces are the common case.
Widen `start_time` and `end_time` to cover when the data was sent, not
only when it was recorded, and those traces come back.

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
    limit = 25 # int | Maximum items to return. Defaults to 25 if omitted; maximum is 50. (optional) (default to 25)
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
 **limit** | **int**| Maximum items to return. Defaults to 25 if omitted; maximum is 50. | [optional] [default to 25]
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

