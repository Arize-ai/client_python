# DatasetExampleUpdate

A dataset example with arbitrary user-defined fields. System-managed fields, except 'id', are excluded for update requests. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | System-assigned unique ID for the example | 

## Example

```python
from arize._generated.api_client.models.dataset_example_update import DatasetExampleUpdate

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetExampleUpdate from a JSON string
dataset_example_update_instance = DatasetExampleUpdate.from_json(json)
# print the JSON string representation of the object
print(DatasetExampleUpdate.to_json())

# convert the object into a dict
dataset_example_update_dict = dataset_example_update_instance.to_dict()
# create an instance of DatasetExampleUpdate from a dict
dataset_example_update_from_dict = DatasetExampleUpdate.from_dict(dataset_example_update_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


