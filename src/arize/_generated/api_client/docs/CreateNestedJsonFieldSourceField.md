# CreateNestedJsonFieldSourceField

A nested JSON field used by a chart plot.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**kind** | **str** | Field kind. Must be &#x60;NESTED_JSON_FIELD&#x60;. | 
**data_type** | [**WidgetSourceDataType**](WidgetSourceDataType.md) | Data type of the field. | 
**path** | **str** | Path to the field within the JSON value. | 

## Example

```python
from arize._generated.api_client.models.create_nested_json_field_source_field import CreateNestedJsonFieldSourceField

# TODO update the JSON string below
json = "{}"
# create an instance of CreateNestedJsonFieldSourceField from a JSON string
create_nested_json_field_source_field_instance = CreateNestedJsonFieldSourceField.from_json(json)
# print the JSON string representation of the object
print(CreateNestedJsonFieldSourceField.to_json())

# convert the object into a dict
create_nested_json_field_source_field_dict = create_nested_json_field_source_field_instance.to_dict()
# create an instance of CreateNestedJsonFieldSourceField from a dict
create_nested_json_field_source_field_from_dict = CreateNestedJsonFieldSourceField.from_dict(create_nested_json_field_source_field_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


