# Dashboard

A dashboard groups charts, statistics, text, and tables to help you monitor and analyze your application data. The response includes published widgets only. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The dashboard ID. | 
**space_id** | **str** | The space ID the dashboard belongs to. | 
**name** | **str** | The dashboard name. | 
**status** | [**DashboardStatus**](DashboardStatus.md) |  | 
**timezone** | **str** | Saved timezone for the dashboard; null means clients use the viewer&#39;s local timezone. | [optional] 
**start_time** | **datetime** | The dashboard start time. | [optional] 
**end_time** | **datetime** | The dashboard end time. | [optional] 
**time_series_data_granularity** | **str** | The dashboard time series data granularity. | [optional] 
**time_range_key** | **str** | The dashboard time range key, such as &#x60;7d&#x60; or &#x60;custom&#x60;. | 
**created_at** | **datetime** | When the dashboard was created. | 
**creator_user_id** | **str** | The ID of the user who created the dashboard. | 
**tags** | [**List[DashboardTagSummary]**](DashboardTagSummary.md) | Tags applied to the dashboard. | 
**widgets** | [**List[WidgetSummary]**](WidgetSummary.md) | Summaries of the published widgets on the dashboard. Full widget configuration is available from &#x60;GET /v2/dashboards/{dashboard_id}/widgets/{widget_id}&#x60; (get_dashboard_widget). | 

## Example

```python
from arize._generated.api_client.models.dashboard import Dashboard

# TODO update the JSON string below
json = "{}"
# create an instance of Dashboard from a JSON string
dashboard_instance = Dashboard.from_json(json)
# print the JSON string representation of the object
print(Dashboard.to_json())

# convert the object into a dict
dashboard_dict = dashboard_instance.to_dict()
# create an instance of Dashboard from a dict
dashboard_from_dict = Dashboard.from_dict(dashboard_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


