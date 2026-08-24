# DeleteEvaluatorVersionsResponse

Result of a DELETE /v2/evaluators/{evaluator_id}/versions request.  The delete is partial-tolerant: requested versions that exist and belong to `evaluator_id` are deleted; every requested ID that was not deleted is reported in `not_deleted_version_ids`. An ID may be not-deleted because it does not exist or belongs to a different evaluator.  `completed` is `true` when this response is returned because the synchronous delete has fully processed the request. It does not mean every requested version was found and deleted: each requested ID appears in exactly one of `deleted_version_ids` or `not_deleted_version_ids`.  The delete operation is idempotent — re-submitting already-deleted IDs is safe and simply reports them as not deleted.  Deleting a version that is currently pinned to a running online task un-pins that task, which then falls back to resolving the evaluator's latest version. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**completed** | **bool** | Always &#x60;true&#x60; in a successful response, indicating both result lists are complete. This does not indicate whether all requested versions existed.  | 
**deleted_version_ids** | **List[str]** | Evaluator version IDs confirmed deleted in this request. | 
**not_deleted_version_ids** | **List[str]** | Requested evaluator version IDs that were not deleted. | 

## Example

```python
from arize._generated.api_client.models.delete_evaluator_versions_response import DeleteEvaluatorVersionsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of DeleteEvaluatorVersionsResponse from a JSON string
delete_evaluator_versions_response_instance = DeleteEvaluatorVersionsResponse.from_json(json)
# print the JSON string representation of the object
print(DeleteEvaluatorVersionsResponse.to_json())

# convert the object into a dict
delete_evaluator_versions_response_dict = delete_evaluator_versions_response_instance.to_dict()
# create an instance of DeleteEvaluatorVersionsResponse from a dict
delete_evaluator_versions_response_from_dict = DeleteEvaluatorVersionsResponse.from_dict(delete_evaluator_versions_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


