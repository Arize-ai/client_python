# EvaluatorsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluators** | [**List[Evaluator]**](Evaluator.md) | A list of evaluators | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluators_list200_response import EvaluatorsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorsList200Response from a JSON string
evaluators_list200_response_instance = EvaluatorsList200Response.from_json(json)
# print the JSON string representation of the object
print(EvaluatorsList200Response.to_json())

# convert the object into a dict
evaluators_list200_response_dict = evaluators_list200_response_instance.to_dict()
# create an instance of EvaluatorsList200Response from a dict
evaluators_list200_response_from_dict = EvaluatorsList200Response.from_dict(evaluators_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


