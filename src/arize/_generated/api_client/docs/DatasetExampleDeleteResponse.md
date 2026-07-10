# DatasetExampleDeleteResponse

Result of a DELETE dataset examples request.  The delete is partial-tolerant: examples that exist in the selected version are deleted, and every requested ID that was not deleted is reported in `not_deleted_example_ids` so the caller can act on it.  A `200 OK` response always includes: - `completed` — `true` if the operation finished and no retry is needed;   `false` if it could not fully complete (retry the full request). - `deleted_example_ids` — example IDs confirmed deleted in this request. - `not_deleted_example_ids` — requested IDs not deleted: either not found in   the selected version (never added, or already deleted), or whose deletion   did not complete when `completed` is `false`. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**completed** | **bool** | &#x60;true&#x60; when the operation finished and no retry is needed. &#x60;false&#x60; when the operation could not fully complete — retry the original full request (the delete is idempotent).  | 
**deleted_example_ids** | **List[str]** | Example IDs confirmed deleted in this request. | 
**not_deleted_example_ids** | **List[str]** | Requested example IDs that were not deleted: either not found in the selected version (never added, or already deleted), or whose deletion did not complete when &#x60;completed&#x60; is &#x60;false&#x60;.  | 

## Example

```python
from arize._generated.api_client.models.dataset_example_delete_response import DatasetExampleDeleteResponse

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetExampleDeleteResponse from a JSON string
dataset_example_delete_response_instance = DatasetExampleDeleteResponse.from_json(json)
# print the JSON string representation of the object
print(DatasetExampleDeleteResponse.to_json())

# convert the object into a dict
dataset_example_delete_response_dict = dataset_example_delete_response_instance.to_dict()
# create an instance of DatasetExampleDeleteResponse from a dict
dataset_example_delete_response_from_dict = DatasetExampleDeleteResponse.from_dict(dataset_example_delete_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


