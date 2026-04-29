# Evaluator

An evaluator defines reusable evaluation logic that can be attached to evaluation tasks. The type field determines the kind of evaluation: template (LLM-based template evaluation) or code (custom code evaluation). 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the evaluator | 
**name** | **str** | The name of the evaluator | 
**description** | **str** | The description of the evaluator | [optional] 
**type** | [**EvaluatorType**](EvaluatorType.md) |  | 
**space_id** | **str** | The unique identifier for the space the evaluator belongs to | 
**created_at** | **datetime** | When the evaluator was created | 
**updated_at** | **datetime** | When the evaluator was last updated | 
**created_by_user_id** | **str** | The unique identifier for the user who created the evaluator | 

## Example

```python
from arize._generated.api_client.models.evaluator import Evaluator

# TODO update the JSON string below
json = "{}"
# create an instance of Evaluator from a JSON string
evaluator_instance = Evaluator.from_json(json)
# print the JSON string representation of the object
print(Evaluator.to_json())

# convert the object into a dict
evaluator_dict = evaluator_instance.to_dict()
# create an instance of Evaluator from a dict
evaluator_from_dict = Evaluator.from_dict(evaluator_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


