# NestedJsonFieldSourceField

A nested field inside a JSON dataset column used by a widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**kind** | [**WidgetSourceFieldKind**](WidgetSourceFieldKind.md) | Identifies this source as a nested JSON field. | 
**data_type** | [**WidgetSourceDataType**](WidgetSourceDataType.md) | Data type of the selected nested field. | 
**path** | **str** | Saved path to the nested field. | 

## Example

```python
from arize._generated.api_client.models.nested_json_field_source_field import NestedJsonFieldSourceField

# TODO update the JSON string below
json = "{}"
# create an instance of NestedJsonFieldSourceField from a JSON string
nested_json_field_source_field_instance = NestedJsonFieldSourceField.from_json(json)
# print the JSON string representation of the object
print(NestedJsonFieldSourceField.to_json())

# convert the object into a dict
nested_json_field_source_field_dict = nested_json_field_source_field_instance.to_dict()
# create an instance of NestedJsonFieldSourceField from a dict
nested_json_field_source_field_from_dict = NestedJsonFieldSourceField.from_dict(nested_json_field_source_field_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


