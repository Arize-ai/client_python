# DatasetExample

A dataset example with arbitrary user-defined fields. System-managed fields are included as read-only for responses. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | System-assigned unique ID for the example | [readonly] 
**created_at** | **datetime** | Timestamp for when the example was created | [readonly] 
**updated_at** | **datetime** | Timestamp for the last update of the example | [readonly] 

## Example

```python
from arize._generated.api_client.models.dataset_example import DatasetExample

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetExample from a JSON string
dataset_example_instance = DatasetExample.from_json(json)
# print the JSON string representation of the object
print(DatasetExample.to_json())

# convert the object into a dict
dataset_example_dict = dataset_example_instance.to_dict()
# create an instance of DatasetExample from a dict
dataset_example_from_dict = DatasetExample.from_dict(dataset_example_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


