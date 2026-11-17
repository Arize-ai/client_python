# CreateTextWidgetRequest

Request body for creating a text widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Widget type. Must be &#x60;TEXT&#x60;. | 
**title** | **str** | Display title of the widget. | 
**grid_position** | **List[int]** | Widget position as [rowStart, colStart, rowEnd, colEnd]. The next available position is used when omitted. | [optional] 
**creation_status** | [**WidgetCreationStatus**](WidgetCreationStatus.md) | Whether the widget is created as a draft or published. Defaults to &#x60;PUBLISHED&#x60;. | [optional] 
**content** | **str** | Markdown content displayed by the widget. | 

## Example

```python
from arize._generated.api_client.models.create_text_widget_request import CreateTextWidgetRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateTextWidgetRequest from a JSON string
create_text_widget_request_instance = CreateTextWidgetRequest.from_json(json)
# print the JSON string representation of the object
print(CreateTextWidgetRequest.to_json())

# convert the object into a dict
create_text_widget_request_dict = create_text_widget_request_instance.to_dict()
# create an instance of CreateTextWidgetRequest from a dict
create_text_widget_request_from_dict = CreateTextWidgetRequest.from_dict(create_text_widget_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


