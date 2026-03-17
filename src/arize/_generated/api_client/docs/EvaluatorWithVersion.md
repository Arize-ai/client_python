# EvaluatorWithVersion


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the evaluator | 
**name** | **str** | The name of the evaluator | 
**description** | **str** | The description of the evaluator | [optional] 
**type** | **str** | The evaluator type: template (LLM-based) or code (custom code) | 
**space_id** | **str** | The unique identifier for the space the evaluator belongs to | 
**created_at** | **datetime** | When the evaluator was created | 
**updated_at** | **datetime** | When the evaluator was last updated | 
**created_by_user_id** | **str** | The unique identifier for the user who created the evaluator | 
**version** | [**EvaluatorVersion**](EvaluatorVersion.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluator_with_version import EvaluatorWithVersion

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorWithVersion from a JSON string
evaluator_with_version_instance = EvaluatorWithVersion.from_json(json)
# print the JSON string representation of the object
print(EvaluatorWithVersion.to_json())

# convert the object into a dict
evaluator_with_version_dict = evaluator_with_version_instance.to_dict()
# create an instance of EvaluatorWithVersion from a dict
evaluator_with_version_from_dict = EvaluatorWithVersion.from_dict(evaluator_with_version_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


