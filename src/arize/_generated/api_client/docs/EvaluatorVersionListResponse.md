# EvaluatorVersionListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluator_versions** | [**List[EvaluatorVersion]**](EvaluatorVersion.md) | A list of evaluator versions | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluator_version_list_response import EvaluatorVersionListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorVersionListResponse from a JSON string
evaluator_version_list_response_instance = EvaluatorVersionListResponse.from_json(json)
# print the JSON string representation of the object
print(EvaluatorVersionListResponse.to_json())

# convert the object into a dict
evaluator_version_list_response_dict = evaluator_version_list_response_instance.to_dict()
# create an instance of EvaluatorVersionListResponse from a dict
evaluator_version_list_response_from_dict = EvaluatorVersionListResponse.from_dict(evaluator_version_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


