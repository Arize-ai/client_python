# EvaluatorVersionCodeCreate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**commit_message** | **str** | Commit message describing the changes | 
**code_config** | [**CodeConfig**](CodeConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluator_version_code_create import EvaluatorVersionCodeCreate

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorVersionCodeCreate from a JSON string
evaluator_version_code_create_instance = EvaluatorVersionCodeCreate.from_json(json)
# print the JSON string representation of the object
print(EvaluatorVersionCodeCreate.to_json())

# convert the object into a dict
evaluator_version_code_create_dict = evaluator_version_code_create_instance.to_dict()
# create an instance of EvaluatorVersionCodeCreate from a dict
evaluator_version_code_create_from_dict = EvaluatorVersionCodeCreate.from_dict(evaluator_version_code_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


