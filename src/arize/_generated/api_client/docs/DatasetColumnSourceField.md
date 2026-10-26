# DatasetColumnSourceField

A top-level dataset column used by a widget.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**kind** | [**WidgetSourceFieldKind**](WidgetSourceFieldKind.md) | Identifies this source as a top-level dataset column. | 
**data_type** | [**WidgetSourceDataType**](WidgetSourceDataType.md) | Data type of the selected column. | 
**column_name** | **str** | Name of the selected dataset column. | 

## Example

```python
from arize._generated.api_client.models.dataset_column_source_field import DatasetColumnSourceField

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetColumnSourceField from a JSON string
dataset_column_source_field_instance = DatasetColumnSourceField.from_json(json)
# print the JSON string representation of the object
print(DatasetColumnSourceField.to_json())

# convert the object into a dict
dataset_column_source_field_dict = dataset_column_source_field_instance.to_dict()
# create an instance of DatasetColumnSourceField from a dict
dataset_column_source_field_from_dict = DatasetColumnSourceField.from_dict(dataset_column_source_field_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


