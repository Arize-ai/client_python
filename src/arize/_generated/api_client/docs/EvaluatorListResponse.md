# EvaluatorListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluators** | [**List[Evaluator]**](Evaluator.md) | A list of evaluators | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluator_list_response import EvaluatorListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorListResponse from a JSON string
evaluator_list_response_instance = EvaluatorListResponse.from_json(json)
# print the JSON string representation of the object
print(EvaluatorListResponse.to_json())

# convert the object into a dict
evaluator_list_response_dict = evaluator_list_response_instance.to_dict()
# create an instance of EvaluatorListResponse from a dict
evaluator_list_response_from_dict = EvaluatorListResponse.from_dict(evaluator_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


