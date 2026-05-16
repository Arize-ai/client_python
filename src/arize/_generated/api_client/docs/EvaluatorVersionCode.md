# EvaluatorVersionCode

Evaluator version carrying a code configuration.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for this version | 
**evaluator_id** | **str** | The parent evaluator ID | 
**commit_hash** | **str** | A unique hash identifying this version | 
**commit_message** | **str** | A message describing the changes in this version | 
**created_at** | **datetime** | When this version was created | 
**created_by_user_id** | **str** | The unique identifier for the user who created this version | 
**type** | **str** | Evaluator version type. Must be &#x60;code&#x60; for code evaluator versions; must match the parent evaluator&#39;s &#x60;type&#x60;. | 
**code_config** | [**CodeConfig**](CodeConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluator_version_code import EvaluatorVersionCode

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorVersionCode from a JSON string
evaluator_version_code_instance = EvaluatorVersionCode.from_json(json)
# print the JSON string representation of the object
print(EvaluatorVersionCode.to_json())

# convert the object into a dict
evaluator_version_code_dict = evaluator_version_code_instance.to_dict()
# create an instance of EvaluatorVersionCode from a dict
evaluator_version_code_from_dict = EvaluatorVersionCode.from_dict(evaluator_version_code_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


