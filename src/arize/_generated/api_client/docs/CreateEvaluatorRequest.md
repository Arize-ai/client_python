# CreateEvaluatorRequest

Body containing evaluator creation parameters with an initial version.  Only `type: TEMPLATE` and `type: CODE` are currently accepted on creation. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**space_id** | **str** | Space identifier (base64) | 
**name** | **str** | Evaluator name (must be unique within the space) | 
**description** | **str** | Evaluator description | [optional] 
**type** | [**EvaluatorType**](EvaluatorType.md) |  | 
**version** | [**CreateEvaluatorVersionRequest**](CreateEvaluatorVersionRequest.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_evaluator_request import CreateEvaluatorRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateEvaluatorRequest from a JSON string
create_evaluator_request_instance = CreateEvaluatorRequest.from_json(json)
# print the JSON string representation of the object
print(CreateEvaluatorRequest.to_json())

# convert the object into a dict
create_evaluator_request_dict = create_evaluator_request_instance.to_dict()
# create an instance of CreateEvaluatorRequest from a dict
create_evaluator_request_from_dict = CreateEvaluatorRequest.from_dict(create_evaluator_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


