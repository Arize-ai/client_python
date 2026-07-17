# UpdateEvaluatorRequest

Body containing evaluator update parameters

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New evaluator name | [optional] 
**description** | **str** | New description | [optional] 

## Example

```python
from arize._generated.api_client.models.update_evaluator_request import UpdateEvaluatorRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateEvaluatorRequest from a JSON string
update_evaluator_request_instance = UpdateEvaluatorRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateEvaluatorRequest.to_json())

# convert the object into a dict
update_evaluator_request_dict = update_evaluator_request_instance.to_dict()
# create an instance of UpdateEvaluatorRequest from a dict
update_evaluator_request_from_dict = UpdateEvaluatorRequest.from_dict(update_evaluator_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


