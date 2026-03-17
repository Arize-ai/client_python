# EvaluatorsUpdateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New evaluator name | [optional] 
**description** | **str** | New description | [optional] 

## Example

```python
from arize._generated.api_client.models.evaluators_update_request import EvaluatorsUpdateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorsUpdateRequest from a JSON string
evaluators_update_request_instance = EvaluatorsUpdateRequest.from_json(json)
# print the JSON string representation of the object
print(EvaluatorsUpdateRequest.to_json())

# convert the object into a dict
evaluators_update_request_dict = evaluators_update_request_instance.to_dict()
# create an instance of EvaluatorsUpdateRequest from a dict
evaluators_update_request_from_dict = EvaluatorsUpdateRequest.from_dict(evaluators_update_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


