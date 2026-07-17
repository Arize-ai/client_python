# CreateEvaluatorVersionRequest

Payload for an evaluator version: exactly one of `template_config` or `code_config`. Used both when creating an evaluator (initial `version`) and when appending a version. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**commit_message** | **str** | Commit message describing the changes | 
**template_config** | [**TemplateConfig**](TemplateConfig.md) |  | 
**code_config** | [**CodeConfig**](CodeConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_evaluator_version_request import CreateEvaluatorVersionRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateEvaluatorVersionRequest from a JSON string
create_evaluator_version_request_instance = CreateEvaluatorVersionRequest.from_json(json)
# print the JSON string representation of the object
print(CreateEvaluatorVersionRequest.to_json())

# convert the object into a dict
create_evaluator_version_request_dict = create_evaluator_version_request_instance.to_dict()
# create an instance of CreateEvaluatorVersionRequest from a dict
create_evaluator_version_request_from_dict = CreateEvaluatorVersionRequest.from_dict(create_evaluator_version_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


