# EvaluatorsCreateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**space_id** | **str** | Space global ID (base64) | 
**name** | **str** | Evaluator name (must be unique within the space) | 
**description** | **str** | Evaluator description | [optional] 
**type** | **str** | Evaluator type. Only template is supported in this iteration. | 
**version** | [**EvaluatorsCreateRequestVersion**](EvaluatorsCreateRequestVersion.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluators_create_request import EvaluatorsCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorsCreateRequest from a JSON string
evaluators_create_request_instance = EvaluatorsCreateRequest.from_json(json)
# print the JSON string representation of the object
print(EvaluatorsCreateRequest.to_json())

# convert the object into a dict
evaluators_create_request_dict = evaluators_create_request_instance.to_dict()
# create an instance of EvaluatorsCreateRequest from a dict
evaluators_create_request_from_dict = EvaluatorsCreateRequest.from_dict(evaluators_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


