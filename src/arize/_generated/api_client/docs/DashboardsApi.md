# arize._generated.api_client.DashboardsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_dashboard**](DashboardsApi.md#create_dashboard) | **POST** /v2/dashboards | Create a dashboard
[**get_dashboard**](DashboardsApi.md#get_dashboard) | **GET** /v2/dashboards/{dashboard_id} | Get a dashboard
[**get_dashboard_widget**](DashboardsApi.md#get_dashboard_widget) | **GET** /v2/dashboards/{dashboard_id}/widgets/{widget_id} | Get a dashboard widget
[**list_dashboard_widgets**](DashboardsApi.md#list_dashboard_widgets) | **GET** /v2/dashboards/{dashboard_id}/widgets | List dashboard widgets
[**list_dashboards**](DashboardsApi.md#list_dashboards) | **GET** /v2/dashboards | List dashboards


# **create_dashboard**
> Dashboard create_dashboard(create_dashboard_request)

Create a dashboard

Create an empty dashboard in a space. The caller must have permission to
create dashboards in the target space. New dashboards start active, and
widgets are added separately.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_dashboard_request import CreateDashboardRequest
from arize._generated.api_client.models.dashboard import Dashboard
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
    api_instance = arize._generated.api_client.DashboardsApi(api_client)
    create_dashboard_request = {"space_id":"U3BhY2U6MTIzNDU=","name":"Production Overview","timezone":"America/Los_Angeles","start_time":"2024-01-01T00:00:00Z","end_time":"2024-01-08T00:00:00Z","time_series_data_granularity":"hour","time_range_key":"7d"} # CreateDashboardRequest | Body containing dashboard creation parameters

    try:
        # Create a dashboard
        api_response = api_instance.create_dashboard(create_dashboard_request)
        print("The response of DashboardsApi->create_dashboard:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DashboardsApi->create_dashboard: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_dashboard_request** | [**CreateDashboardRequest**](CreateDashboardRequest.md)| Body containing dashboard creation parameters | 

### Return type

[**Dashboard**](Dashboard.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Dashboard created |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_dashboard**
> Dashboard get_dashboard(dashboard_id)

Get a dashboard

Get a dashboard by its ID. The dashboard includes its tags and published widget summaries.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.dashboard import Dashboard
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
    api_instance = arize._generated.api_client.DashboardsApi(api_client)
    dashboard_id = 'RGFzaGJvYXJkOjEyMzQ1' # str | The Relay global ID of the dashboard.

    try:
        # Get a dashboard
        api_response = api_instance.get_dashboard(dashboard_id)
        print("The response of DashboardsApi->get_dashboard:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DashboardsApi->get_dashboard: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dashboard_id** | **str**| The Relay global ID of the dashboard. | 

### Return type

[**Dashboard**](Dashboard.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a dashboard with its tags and published widgets. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_dashboard_widget**
> Widget get_dashboard_widget(dashboard_id, widget_id)

Get a dashboard widget

Get a widget, including its saved configuration, from a dashboard. Draft and
published widgets can be read. The widget must belong to the specified
dashboard, and the caller must have permission to read that dashboard.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.widget import Widget
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
    api_instance = arize._generated.api_client.DashboardsApi(api_client)
    dashboard_id = 'RGFzaGJvYXJkOjEyMzQ1' # str | The Relay global ID of the dashboard.
    widget_id = 'TGluZUNoYXJ0V2lkZ2V0OjEyMzQ1' # str | The Relay global ID of the widget.

    try:
        # Get a dashboard widget
        api_response = api_instance.get_dashboard_widget(dashboard_id, widget_id)
        print("The response of DashboardsApi->get_dashboard_widget:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DashboardsApi->get_dashboard_widget: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dashboard_id** | **str**| The Relay global ID of the dashboard. | 
 **widget_id** | **str**| The Relay global ID of the widget. | 

### Return type

[**Widget**](Widget.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns one dashboard widget with its type-specific configuration. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_dashboard_widgets**
> ListDashboardWidgetsResponse list_dashboard_widgets(dashboard_id, type=type, creation_status=creation_status, limit=limit, cursor=cursor)

List dashboard widgets

List the widgets on a dashboard as summaries (id, title, type, and parent
dashboard). Use [Get a dashboard widget](#tag/Dashboards/operation/get_dashboard_widget)
to retrieve a widget's full configuration. The caller must have permission to
read the dashboard.

Only published widgets are returned by default. Set `creation_status` to
`DRAFT` to list draft widgets instead. Use `type` to return only widgets of one
type. Results use cursor-based pagination.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_dashboard_widgets_response import ListDashboardWidgetsResponse
from arize._generated.api_client.models.widget_creation_status import WidgetCreationStatus
from arize._generated.api_client.models.widget_type import WidgetType
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
    api_instance = arize._generated.api_client.DashboardsApi(api_client)
    dashboard_id = 'RGFzaGJvYXJkOjEyMzQ1' # str | The Relay global ID of the dashboard.
    type = arize._generated.api_client.WidgetType() # WidgetType | Filter widgets by type, for example `LINE_CHART`. If omitted, widgets of every type are returned. (optional)
    creation_status = arize._generated.api_client.WidgetCreationStatus() # WidgetCreationStatus | Filter widgets by creation status, either `DRAFT` or `PUBLISHED`. Defaults to `PUBLISHED` if omitted. (optional)
    limit = 50 # int | Maximum items to return. Defaults to 50 if omitted; maximum is 100. (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List dashboard widgets
        api_response = api_instance.list_dashboard_widgets(dashboard_id, type=type, creation_status=creation_status, limit=limit, cursor=cursor)
        print("The response of DashboardsApi->list_dashboard_widgets:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DashboardsApi->list_dashboard_widgets: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dashboard_id** | **str**| The Relay global ID of the dashboard. | 
 **type** | [**WidgetType**](.md)| Filter widgets by type, for example &#x60;LINE_CHART&#x60;. If omitted, widgets of every type are returned. | [optional] 
 **creation_status** | [**WidgetCreationStatus**](.md)| Filter widgets by creation status, either &#x60;DRAFT&#x60; or &#x60;PUBLISHED&#x60;. Defaults to &#x60;PUBLISHED&#x60; if omitted. | [optional] 
 **limit** | **int**| Maximum items to return. Defaults to 50 if omitted; maximum is 100. | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListDashboardWidgetsResponse**](ListDashboardWidgetsResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of widget summaries for the dashboard. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_dashboards**
> ListDashboardsResponse list_dashboards(space_id=space_id, space_name=space_name, name=name, status=status, sort_column=sort_column, sort_dir=sort_dir, limit=limit, cursor=cursor)

List dashboards

List the dashboards the user has access to, including their tags but not
their widgets. Use [Get a dashboard](#tag/Dashboards/operation/get_dashboard)
to retrieve a dashboard's widgets. Only dashboards the caller has permission
to read are returned.

Set `space_id` to list the dashboards in a single space. If omitted,
dashboards are listed across all spaces the caller can read. Use
`space_name` to narrow results to spaces whose name contains the given string.

Only active dashboards are returned by default. Set `status` to `INACTIVE` to
list inactive dashboards instead. Use `name` to search by name, and `sort_column`
and `sort_dir` to control ordering. Results use cursor-based pagination.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_dashboards_response import ListDashboardsResponse
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
    api_instance = arize._generated.api_client.DashboardsApi(api_client)
    space_id = 'U3BhY2U6MTIzNDU=' # str | Filter search results to a particular space ID (optional)
    space_name = 'my-space' # str | Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  (optional)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    status = ACTIVE # str | Filter dashboards by status, either `ACTIVE` or `INACTIVE`. Defaults to `ACTIVE` if omitted. Deleted dashboards are never returned.  (optional) (default to ACTIVE)
    sort_column = NAME # str | The column to sort dashboards by, either `NAME` or `CREATED_AT`. Defaults to `NAME` if omitted. (optional) (default to NAME)
    sort_dir = ASC # str | The sort direction, either `ASC` or `DESC`. Defaults to `ASC` if omitted. (optional) (default to ASC)
    limit = 50 # int | Maximum items to return. Defaults to 50 if omitted; maximum is 100. (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List dashboards
        api_response = api_instance.list_dashboards(space_id=space_id, space_name=space_name, name=name, status=status, sort_column=sort_column, sort_dir=sort_dir, limit=limit, cursor=cursor)
        print("The response of DashboardsApi->list_dashboards:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DashboardsApi->list_dashboards: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| Filter search results to a particular space ID | [optional] 
 **space_name** | **str**| Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. If omitted, no name filtering is applied and all resources are returned.  | [optional] 
 **status** | **str**| Filter dashboards by status, either &#x60;ACTIVE&#x60; or &#x60;INACTIVE&#x60;. Defaults to &#x60;ACTIVE&#x60; if omitted. Deleted dashboards are never returned.  | [optional] [default to ACTIVE]
 **sort_column** | **str**| The column to sort dashboards by, either &#x60;NAME&#x60; or &#x60;CREATED_AT&#x60;. Defaults to &#x60;NAME&#x60; if omitted. | [optional] [default to NAME]
 **sort_dir** | **str**| The sort direction, either &#x60;ASC&#x60; or &#x60;DESC&#x60;. Defaults to &#x60;ASC&#x60; if omitted. | [optional] [default to ASC]
 **limit** | **int**| Maximum items to return. Defaults to 50 if omitted; maximum is 100. | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListDashboardsResponse**](ListDashboardsResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of dashboards in the space. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

