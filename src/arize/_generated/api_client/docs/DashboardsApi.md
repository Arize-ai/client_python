# arize._generated.api_client.DashboardsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_dashboard_widget**](DashboardsApi.md#get_dashboard_widget) | **GET** /v2/dashboards/{dashboard_id}/widgets/{widget_id} | Get a dashboard widget


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

