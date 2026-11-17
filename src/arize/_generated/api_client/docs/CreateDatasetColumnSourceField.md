# CreateDatasetColumnSourceField

A dataset column used by a chart plot.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**kind** | **str** | Field kind. Must be &#x60;DATASET_COLUMN&#x60;. | 
**data_type** | [**WidgetSourceDataType**](WidgetSourceDataType.md) | Data type of the column. | 
**column_name** | **str** | Name of the dataset column. | 

## Example

```python
from arize._generated.api_client.models.create_dataset_column_source_field import CreateDatasetColumnSourceField

# TODO update the JSON string below
json = "{}"
# create an instance of CreateDatasetColumnSourceField from a JSON string
create_dataset_column_source_field_instance = CreateDatasetColumnSourceField.from_json(json)
# print the JSON string representation of the object
print(CreateDatasetColumnSourceField.to_json())

# convert the object into a dict
create_dataset_column_source_field_dict = create_dataset_column_source_field_instance.to_dict()
# create an instance of CreateDatasetColumnSourceField from a dict
create_dataset_column_source_field_from_dict = CreateDatasetColumnSourceField.from_dict(create_dataset_column_source_field_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


