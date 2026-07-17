# UpdateDatasetExampleInput

A dataset example with arbitrary user-defined fields. System-managed fields, except 'id', are excluded for update requests. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | System-assigned unique ID for the example | 

## Example

```python
from arize._generated.api_client.models.update_dataset_example_input import UpdateDatasetExampleInput

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateDatasetExampleInput from a JSON string
update_dataset_example_input_instance = UpdateDatasetExampleInput.from_json(json)
# print the JSON string representation of the object
print(UpdateDatasetExampleInput.to_json())

# convert the object into a dict
update_dataset_example_input_dict = update_dataset_example_input_instance.to_dict()
# create an instance of UpdateDatasetExampleInput from a dict
update_dataset_example_input_from_dict = UpdateDatasetExampleInput.from_dict(update_dataset_example_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


