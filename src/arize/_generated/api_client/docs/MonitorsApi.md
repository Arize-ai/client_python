# arize._generated.api_client.MonitorsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_monitor**](MonitorsApi.md#get_monitor) | **GET** /v2/monitors/{monitor_id} | Get a monitor
[**list_monitors**](MonitorsApi.md#list_monitors) | **GET** /v2/monitors | List monitors


# **get_monitor**
> Monitor get_monitor(monitor_id)

Get a monitor

Get a monitor by its ID.

The response shape varies by `type` (`data_quality`, `performance`,
`drift`, `custom_metric`, `tracing`)

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.monitor import Monitor
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
    api_instance = arize._generated.api_client.MonitorsApi(api_client)
    monitor_id = 'TW9uaXRvcjoxMjM=' # str | The unique monitor identifier (base64)

    try:
        # Get a monitor
        api_response = api_instance.get_monitor(monitor_id)
        print("The response of MonitorsApi->get_monitor:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling MonitorsApi->get_monitor: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **monitor_id** | **str**| The unique monitor identifier (base64) | 

### Return type

[**Monitor**](Monitor.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a single monitor object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_monitors**
> ListMonitorsResponse list_monitors(space_id=space_id, space_name=space_name, name=name, project_id=project_id, project_name=project_name, type=type, status=status, notifications_enabled=notifications_enabled, dimension_category=dimension_category, dimension_name=dimension_name, data_quality_metric=data_quality_metric, performance_metric=performance_metric, drift_metric=drift_metric, custom_metric_id=custom_metric_id, limit=limit, cursor=cursor)

List monitors

List monitors the caller can read, with filtering and cursor-based
pagination. Results are ordered by creation time, newest first.
Deleted and draft monitors are excluded.

The shape of each returned monitor varies by `type` (`DATA_QUALITY`,
`PERFORMANCE`, `DRIFT`, `CUSTOM_METRIC`, `TRACING`).

All filters are optional and compose with AND semantics (a monitor
must match every provided filter). When a filter is omitted, no
filtering is applied for that field. Filters that cannot be satisfied
together — a metric filter from one family combined with a `type` from
another, or two metric filters from different families — are valid
input and return a `200` with an empty `monitors` array, as does any
other combination that simply matches nothing.

The four name filters do not all match the same way. `name` and
`space_name` are case-insensitive substring searches, so `name=prod`
matches "production". `project_name` and `dimension_name` are exact,
case-sensitive matches, so they need the full name as stored — for
`dimension_name`, the value copied verbatim from a returned monitor's
`dimension.name`.

An identifier that is not a well-formed ID of the expected kind
returns a `400`. A well-formed `space_id`, `project_id`, or
`custom_metric_id` that either does not exist or is not readable by
the caller returns the same `404` in both cases, so the response never
reveals whether the referenced resource exists.

A caller whose credentials grant monitor read access in no space at
all receives a `403`.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.data_quality_metric import DataQualityMetric
from arize._generated.api_client.models.dimension_category import DimensionCategory
from arize._generated.api_client.models.drift_metric import DriftMetric
from arize._generated.api_client.models.list_monitors_response import ListMonitorsResponse
from arize._generated.api_client.models.monitor_status import MonitorStatus
from arize._generated.api_client.models.monitor_type import MonitorType
from arize._generated.api_client.models.performance_metric import PerformanceMetric
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
    api_instance = arize._generated.api_client.MonitorsApi(api_client)
    space_id = 'U3BhY2U6MTIzNDU=' # str | Filter search results to a particular space ID (optional)
    space_name = 'my-space' # str | Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  (optional)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    project_id = 'TW9kZWw6MTIzOmFCY0Q=' # str | Filter results to resources associated with a specific project (base64 identifier). If omitted, results are not filtered by project.  (optional)
    project_name = 'my-llm-app' # str | Exact-match filter on the name of the project the monitor's primary metric is computed over. Unlike `name` and `space_name`, this is an exact (case-sensitive) match, not a substring search. If omitted, no project name filtering is applied.  (optional)
    type = arize._generated.api_client.MonitorType() # MonitorType | Filter by monitor type. Types are exact: `DATA_QUALITY` does not include `TRACING` monitors, and `PERFORMANCE` does not include `CUSTOM_METRIC` monitors. If omitted, monitors of all types are returned.  (optional)
    status = arize._generated.api_client.MonitorStatus() # MonitorStatus | Filter by the monitor's current evaluation state (`TRIGGERED`, `CLEARED`, or `NO_DATA`). If omitted, monitors in every state are returned.  (optional)
    notifications_enabled = true # bool | Filter by whether notifications fire on a triggered transition. `true` returns only monitors with notifications enabled; `false` returns only monitors with notifications disabled. If omitted, monitors are returned regardless of notification state.  (optional)
    dimension_category = arize._generated.api_client.DimensionCategory() # DimensionCategory | Filter to monitors whose metric is computed over a dimension of this category. Values copied from a returned monitor's `dimension.category` work as filters. If omitted, no dimension category filtering is applied.  (optional)
    dimension_name = 'eval.Hallucination.label' # str | Exact-match filter on the name of the dimension the monitor's metric is computed over. Values copied from a returned monitor's `dimension.name` work as filters. If omitted, no dimension name filtering is applied.  (optional)
    data_quality_metric = arize._generated.api_client.DataQualityMetric() # DataQualityMetric | Filter to monitors computing this data quality metric. Matches both `DATA_QUALITY` and `TRACING` monitors; combine with `type` to narrow to one of them. If omitted, no data quality metric filtering is applied.  (optional)
    performance_metric = arize._generated.api_client.PerformanceMetric() # PerformanceMetric | Filter to `PERFORMANCE` monitors computing this performance metric. Does not match `CUSTOM_METRIC` monitors. If omitted, no performance metric filtering is applied.  (optional)
    drift_metric = arize._generated.api_client.DriftMetric() # DriftMetric | Filter to `DRIFT` monitors computing this drift metric. If omitted, no drift metric filtering is applied.  (optional)
    custom_metric_id = 'Q3VzdG9tTWV0cmljOjEyMzQ1' # str | Filter to `CUSTOM_METRIC` monitors evaluating this custom metric (base64 identifier). If omitted, no custom metric filtering is applied.  (optional)
    limit = 50 # int | Maximum items to return. Defaults to 50 if omitted; maximum is 100. (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List monitors
        api_response = api_instance.list_monitors(space_id=space_id, space_name=space_name, name=name, project_id=project_id, project_name=project_name, type=type, status=status, notifications_enabled=notifications_enabled, dimension_category=dimension_category, dimension_name=dimension_name, data_quality_metric=data_quality_metric, performance_metric=performance_metric, drift_metric=drift_metric, custom_metric_id=custom_metric_id, limit=limit, cursor=cursor)
        print("The response of MonitorsApi->list_monitors:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling MonitorsApi->list_monitors: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| Filter search results to a particular space ID | [optional] 
 **space_name** | **str**| Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. If omitted, no name filtering is applied and all resources are returned.  | [optional] 
 **project_id** | **str**| Filter results to resources associated with a specific project (base64 identifier). If omitted, results are not filtered by project.  | [optional] 
 **project_name** | **str**| Exact-match filter on the name of the project the monitor&#39;s primary metric is computed over. Unlike &#x60;name&#x60; and &#x60;space_name&#x60;, this is an exact (case-sensitive) match, not a substring search. If omitted, no project name filtering is applied.  | [optional] 
 **type** | [**MonitorType**](.md)| Filter by monitor type. Types are exact: &#x60;DATA_QUALITY&#x60; does not include &#x60;TRACING&#x60; monitors, and &#x60;PERFORMANCE&#x60; does not include &#x60;CUSTOM_METRIC&#x60; monitors. If omitted, monitors of all types are returned.  | [optional] 
 **status** | [**MonitorStatus**](.md)| Filter by the monitor&#39;s current evaluation state (&#x60;TRIGGERED&#x60;, &#x60;CLEARED&#x60;, or &#x60;NO_DATA&#x60;). If omitted, monitors in every state are returned.  | [optional] 
 **notifications_enabled** | **bool**| Filter by whether notifications fire on a triggered transition. &#x60;true&#x60; returns only monitors with notifications enabled; &#x60;false&#x60; returns only monitors with notifications disabled. If omitted, monitors are returned regardless of notification state.  | [optional] 
 **dimension_category** | [**DimensionCategory**](.md)| Filter to monitors whose metric is computed over a dimension of this category. Values copied from a returned monitor&#39;s &#x60;dimension.category&#x60; work as filters. If omitted, no dimension category filtering is applied.  | [optional] 
 **dimension_name** | **str**| Exact-match filter on the name of the dimension the monitor&#39;s metric is computed over. Values copied from a returned monitor&#39;s &#x60;dimension.name&#x60; work as filters. If omitted, no dimension name filtering is applied.  | [optional] 
 **data_quality_metric** | [**DataQualityMetric**](.md)| Filter to monitors computing this data quality metric. Matches both &#x60;DATA_QUALITY&#x60; and &#x60;TRACING&#x60; monitors; combine with &#x60;type&#x60; to narrow to one of them. If omitted, no data quality metric filtering is applied.  | [optional] 
 **performance_metric** | [**PerformanceMetric**](.md)| Filter to &#x60;PERFORMANCE&#x60; monitors computing this performance metric. Does not match &#x60;CUSTOM_METRIC&#x60; monitors. If omitted, no performance metric filtering is applied.  | [optional] 
 **drift_metric** | [**DriftMetric**](.md)| Filter to &#x60;DRIFT&#x60; monitors computing this drift metric. If omitted, no drift metric filtering is applied.  | [optional] 
 **custom_metric_id** | **str**| Filter to &#x60;CUSTOM_METRIC&#x60; monitors evaluating this custom metric (base64 identifier). If omitted, no custom metric filtering is applied.  | [optional] 
 **limit** | **int**| Maximum items to return. Defaults to 50 if omitted; maximum is 100. | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListMonitorsResponse**](ListMonitorsResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of monitor objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

