# EvaluatorVersionsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluator_versions** | [**List[EvaluatorVersion]**](EvaluatorVersion.md) | A list of evaluator versions | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluator_versions_list200_response import EvaluatorVersionsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorVersionsList200Response from a JSON string
evaluator_versions_list200_response_instance = EvaluatorVersionsList200Response.from_json(json)
# print the JSON string representation of the object
print(EvaluatorVersionsList200Response.to_json())

# convert the object into a dict
evaluator_versions_list200_response_dict = evaluator_versions_list200_response_instance.to_dict()
# create an instance of EvaluatorVersionsList200Response from a dict
evaluator_versions_list200_response_from_dict = EvaluatorVersionsList200Response.from_dict(evaluator_versions_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


