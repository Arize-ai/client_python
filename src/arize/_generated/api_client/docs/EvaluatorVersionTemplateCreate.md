# EvaluatorVersionTemplateCreate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**commit_message** | **str** | Commit message describing the changes | 
**template_config** | [**TemplateConfig**](TemplateConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluator_version_template_create import EvaluatorVersionTemplateCreate

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorVersionTemplateCreate from a JSON string
evaluator_version_template_create_instance = EvaluatorVersionTemplateCreate.from_json(json)
# print the JSON string representation of the object
print(EvaluatorVersionTemplateCreate.to_json())

# convert the object into a dict
evaluator_version_template_create_dict = evaluator_version_template_create_instance.to_dict()
# create an instance of EvaluatorVersionTemplateCreate from a dict
evaluator_version_template_create_from_dict = EvaluatorVersionTemplateCreate.from_dict(evaluator_version_template_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


