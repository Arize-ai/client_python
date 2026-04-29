# EvaluatorVersionCreate

Payload for an evaluator version: exactly one of `template_config` or `code_config`. Used both when creating an evaluator (initial `version`) and when appending a version. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**commit_message** | **str** | Commit message describing the changes | 
**template_config** | [**TemplateConfig**](TemplateConfig.md) |  | 
**code_config** | [**CodeConfig**](CodeConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluator_version_create import EvaluatorVersionCreate

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorVersionCreate from a JSON string
evaluator_version_create_instance = EvaluatorVersionCreate.from_json(json)
# print the JSON string representation of the object
print(EvaluatorVersionCreate.to_json())

# convert the object into a dict
evaluator_version_create_dict = evaluator_version_create_instance.to_dict()
# create an instance of EvaluatorVersionCreate from a dict
evaluator_version_create_from_dict = EvaluatorVersionCreate.from_dict(evaluator_version_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


