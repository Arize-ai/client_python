# WidgetSummary

A brief description of a dashboard widget, without its saved configuration.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | A universally unique identifier (base64-encoded opaque string). | 
**title** | **str** | Display title of the widget. | 
**type** | [**WidgetType**](WidgetType.md) |  | 
**grid_position** | **List[int]** | Widget position and size as [x, y, width, height]. | 

## Example

```python
from arize._generated.api_client.models.widget_summary import WidgetSummary

# TODO update the JSON string below
json = "{}"
# create an instance of WidgetSummary from a JSON string
widget_summary_instance = WidgetSummary.from_json(json)
# print the JSON string representation of the object
print(WidgetSummary.to_json())

# convert the object into a dict
widget_summary_dict = widget_summary_instance.to_dict()
# create an instance of WidgetSummary from a dict
widget_summary_from_dict = WidgetSummary.from_dict(widget_summary_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


