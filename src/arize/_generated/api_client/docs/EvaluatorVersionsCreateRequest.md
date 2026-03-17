# EvaluatorVersionsCreateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**commit_message** | **str** | Commit message describing the changes | 
**template_config** | [**TemplateConfig**](TemplateConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluator_versions_create_request import EvaluatorVersionsCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorVersionsCreateRequest from a JSON string
evaluator_versions_create_request_instance = EvaluatorVersionsCreateRequest.from_json(json)
# print the JSON string representation of the object
print(EvaluatorVersionsCreateRequest.to_json())

# convert the object into a dict
evaluator_versions_create_request_dict = evaluator_versions_create_request_instance.to_dict()
# create an instance of EvaluatorVersionsCreateRequest from a dict
evaluator_versions_create_request_from_dict = EvaluatorVersionsCreateRequest.from_dict(evaluator_versions_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


