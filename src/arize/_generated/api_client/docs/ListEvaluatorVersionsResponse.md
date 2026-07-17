# ListEvaluatorVersionsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluator_versions** | [**List[EvaluatorVersion]**](EvaluatorVersion.md) | A list of evaluator versions | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_evaluator_versions_response import ListEvaluatorVersionsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListEvaluatorVersionsResponse from a JSON string
list_evaluator_versions_response_instance = ListEvaluatorVersionsResponse.from_json(json)
# print the JSON string representation of the object
print(ListEvaluatorVersionsResponse.to_json())

# convert the object into a dict
list_evaluator_versions_response_dict = list_evaluator_versions_response_instance.to_dict()
# create an instance of ListEvaluatorVersionsResponse from a dict
list_evaluator_versions_response_from_dict = ListEvaluatorVersionsResponse.from_dict(list_evaluator_versions_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


