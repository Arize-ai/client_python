# UpdateDashboardRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | The dashboard name. Leading and trailing whitespace is trimmed, and the trimmed name must be 1-255 characters. | [optional] 
**timezone** | **str** | The timezone used to display dashboard data. Set to &#x60;null&#x60; to clear it. | [optional] 
**start_time** | **datetime** | The start time for the dashboard&#39;s default time range, as an RFC 3339 timestamp. Set to &#x60;null&#x60; to clear it. Required when the resulting &#x60;time_range_key&#x60; is &#x60;custom&#x60;. | [optional] 
**end_time** | **datetime** | The end time for the dashboard&#39;s default time range, as an RFC 3339 timestamp. Set to &#x60;null&#x60; to clear it. Required when the resulting &#x60;time_range_key&#x60; is &#x60;custom&#x60;, and must not be earlier than &#x60;start_time&#x60;. | [optional] 
**time_series_data_granularity** | **str** | The granularity used to display time series data. Set to &#x60;null&#x60; to clear it. | [optional] 
**time_range_key** | **str** | The dashboard&#39;s default time range. One of &#x60;15m&#x60;, &#x60;1h&#x60;, &#x60;12h&#x60;, &#x60;1d&#x60;, &#x60;7d&#x60;, &#x60;14d&#x60;, &#x60;1mo&#x60;, &#x60;3mo&#x60;, &#x60;6mo&#x60;, or &#x60;custom&#x60;. | [optional] 

## Example

```python
from arize._generated.api_client.models.update_dashboard_request import UpdateDashboardRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateDashboardRequest from a JSON string
update_dashboard_request_instance = UpdateDashboardRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateDashboardRequest.to_json())

# convert the object into a dict
update_dashboard_request_dict = update_dashboard_request_instance.to_dict()
# create an instance of UpdateDashboardRequest from a dict
update_dashboard_request_from_dict = UpdateDashboardRequest.from_dict(update_dashboard_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


