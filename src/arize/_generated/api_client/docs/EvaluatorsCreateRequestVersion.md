# EvaluatorsCreateRequestVersion

The initial version for the evaluator

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**commit_message** | **str** | Commit message for the initial version | 
**template_config** | [**TemplateConfig**](TemplateConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluators_create_request_version import EvaluatorsCreateRequestVersion

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorsCreateRequestVersion from a JSON string
evaluators_create_request_version_instance = EvaluatorsCreateRequestVersion.from_json(json)
# print the JSON string representation of the object
print(EvaluatorsCreateRequestVersion.to_json())

# convert the object into a dict
evaluators_create_request_version_dict = evaluators_create_request_version_instance.to_dict()
# create an instance of EvaluatorsCreateRequestVersion from a dict
evaluators_create_request_version_from_dict = EvaluatorsCreateRequestVersion.from_dict(evaluators_create_request_version_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


