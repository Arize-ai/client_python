# DeleteDatasetExamplesRequest

Body containing the IDs of dataset examples to delete

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dataset_version_id** | **str** | Version to delete the examples from. Required. Examples are removed in place from this version; no new version is created.  | 
**example_ids** | **List[str]** | IDs of the examples to delete. Up to 1000 per request. | 

## Example

```python
from arize._generated.api_client.models.delete_dataset_examples_request import DeleteDatasetExamplesRequest

# TODO update the JSON string below
json = "{}"
# create an instance of DeleteDatasetExamplesRequest from a JSON string
delete_dataset_examples_request_instance = DeleteDatasetExamplesRequest.from_json(json)
# print the JSON string representation of the object
print(DeleteDatasetExamplesRequest.to_json())

# convert the object into a dict
delete_dataset_examples_request_dict = delete_dataset_examples_request_instance.to_dict()
# create an instance of DeleteDatasetExamplesRequest from a dict
delete_dataset_examples_request_from_dict = DeleteDatasetExamplesRequest.from_dict(delete_dataset_examples_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


