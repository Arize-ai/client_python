# WidgetSourceField

A dataset column or nested JSON field used by a widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**kind** | [**WidgetSourceFieldKind**](WidgetSourceFieldKind.md) | Identifies this source as a nested JSON field. | 
**data_type** | [**WidgetSourceDataType**](WidgetSourceDataType.md) | Data type of the selected nested field. | 
**column_name** | **str** | Name of the selected dataset column. | 
**path** | **str** | Saved path to the nested field. | 

## Example

```python
from arize._generated.api_client.models.widget_source_field import WidgetSourceField

# TODO update the JSON string below
json = "{}"
# create an instance of WidgetSourceField from a JSON string
widget_source_field_instance = WidgetSourceField.from_json(json)
# print the JSON string representation of the object
print(WidgetSourceField.to_json())

# convert the object into a dict
widget_source_field_dict = widget_source_field_instance.to_dict()
# create an instance of WidgetSourceField from a dict
widget_source_field_from_dict = WidgetSourceField.from_dict(widget_source_field_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


