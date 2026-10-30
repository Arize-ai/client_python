# ListDashboardWidgetsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**widgets** | [**List[WidgetSummary]**](WidgetSummary.md) | A list of widget summaries. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_dashboard_widgets_response import ListDashboardWidgetsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListDashboardWidgetsResponse from a JSON string
list_dashboard_widgets_response_instance = ListDashboardWidgetsResponse.from_json(json)
# print the JSON string representation of the object
print(ListDashboardWidgetsResponse.to_json())

# convert the object into a dict
list_dashboard_widgets_response_dict = list_dashboard_widgets_response_instance.to_dict()
# create an instance of ListDashboardWidgetsResponse from a dict
list_dashboard_widgets_response_from_dict = ListDashboardWidgetsResponse.from_dict(list_dashboard_widgets_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


