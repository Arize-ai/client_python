# DeleteEvaluatorVersionsRequest

Body identifying the versions to delete from the evaluator named by the `evaluator_id` path parameter. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**version_ids** | **List[str]** | IDs of the evaluator versions to delete (up to 100 per request). IDs that do not belong to &#x60;evaluator_id&#x60; are reported as not deleted. Duplicate IDs are accepted and silently collapsed so each version is processed at most once.  | 

## Example

```python
from arize._generated.api_client.models.delete_evaluator_versions_request import DeleteEvaluatorVersionsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of DeleteEvaluatorVersionsRequest from a JSON string
delete_evaluator_versions_request_instance = DeleteEvaluatorVersionsRequest.from_json(json)
# print the JSON string representation of the object
print(DeleteEvaluatorVersionsRequest.to_json())

# convert the object into a dict
delete_evaluator_versions_request_dict = delete_evaluator_versions_request_instance.to_dict()
# create an instance of DeleteEvaluatorVersionsRequest from a dict
delete_evaluator_versions_request_from_dict = DeleteEvaluatorVersionsRequest.from_dict(delete_evaluator_versions_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


