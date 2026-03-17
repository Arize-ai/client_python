# EvaluatorVersion

A versioned snapshot of an evaluator's configuration.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for this version | 
**evaluator_id** | **str** | The parent evaluator ID | 
**commit_hash** | **str** | A unique hash identifying this version | 
**commit_message** | **str** | A message describing the changes in this version | 
**template_config** | [**TemplateConfig**](TemplateConfig.md) |  | 
**created_at** | **datetime** | When this version was created | 
**created_by_user_id** | **str** | The unique identifier for the user who created this version | 

## Example

```python
from arize._generated.api_client.models.evaluator_version import EvaluatorVersion

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorVersion from a JSON string
evaluator_version_instance = EvaluatorVersion.from_json(json)
# print the JSON string representation of the object
print(EvaluatorVersion.to_json())

# convert the object into a dict
evaluator_version_dict = evaluator_version_instance.to_dict()
# create an instance of EvaluatorVersion from a dict
evaluator_version_from_dict = EvaluatorVersion.from_dict(evaluator_version_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


