# ListEvaluatorsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluators** | [**List[Evaluator]**](Evaluator.md) | A list of evaluators | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_evaluators_response import ListEvaluatorsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListEvaluatorsResponse from a JSON string
list_evaluators_response_instance = ListEvaluatorsResponse.from_json(json)
# print the JSON string representation of the object
print(ListEvaluatorsResponse.to_json())

# convert the object into a dict
list_evaluators_response_dict = list_evaluators_response_instance.to_dict()
# create an instance of ListEvaluatorsResponse from a dict
list_evaluators_response_from_dict = ListEvaluatorsResponse.from_dict(list_evaluators_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


