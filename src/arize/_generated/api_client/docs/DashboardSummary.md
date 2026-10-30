# DashboardSummary

A dashboard without its widgets. Use `GET /v2/dashboards/{dashboard_id}` (get_dashboard) to retrieve a dashboard's widgets. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The dashboard ID. | 
**space_id** | **str** | The space ID the dashboard belongs to. | 
**name** | **str** | The dashboard name. | 
**status** | [**DashboardStatus**](DashboardStatus.md) |  | 
**timezone** | **str** | The dashboard timezone. | [optional] 
**start_time** | **datetime** | The dashboard start time. | [optional] 
**end_time** | **datetime** | The dashboard end time. | [optional] 
**time_series_data_granularity** | **str** | The dashboard time series data granularity. | [optional] 
**time_range_key** | **str** | The dashboard time range key, such as &#x60;7d&#x60; or &#x60;custom&#x60;. | 
**created_at** | **datetime** | When the dashboard was created. | 
**creator_user_id** | **str** | The ID of the user who created the dashboard. | 
**tags** | [**List[DashboardTagSummary]**](DashboardTagSummary.md) | Tags applied to the dashboard. | 

## Example

```python
from arize._generated.api_client.models.dashboard_summary import DashboardSummary

# TODO update the JSON string below
json = "{}"
# create an instance of DashboardSummary from a JSON string
dashboard_summary_instance = DashboardSummary.from_json(json)
# print the JSON string representation of the object
print(DashboardSummary.to_json())

# convert the object into a dict
dashboard_summary_dict = dashboard_summary_instance.to_dict()
# create an instance of DashboardSummary from a dict
dashboard_summary_from_dict = DashboardSummary.from_dict(dashboard_summary_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


