# TextWidget

A dashboard widget that displays formatted text.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the widget. | 
**type** | [**WidgetType**](WidgetType.md) | Identifies this widget as a text widget. | 
**dashboard_id** | **str** | Unique identifier for the dashboard containing the widget. | 
**title** | **str** | Display title of the widget. | 
**grid_position** | **List[int]** | Position and size of the widget in the dashboard grid. | 
**creation_status** | [**WidgetCreationStatus**](WidgetCreationStatus.md) | Publication state of the widget. | 
**created_at** | **datetime** | Timestamp when the widget was created. | 
**creator_user_id** | **str** | Unique identifier for the user who created the widget. | 
**content** | **str** | Saved text content displayed by the widget. | 

## Example

```python
from arize._generated.api_client.models.text_widget import TextWidget

# TODO update the JSON string below
json = "{}"
# create an instance of TextWidget from a JSON string
text_widget_instance = TextWidget.from_json(json)
# print the JSON string representation of the object
print(TextWidget.to_json())

# convert the object into a dict
text_widget_dict = text_widget_instance.to_dict()
# create an instance of TextWidget from a dict
text_widget_from_dict = TextWidget.from_dict(text_widget_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


