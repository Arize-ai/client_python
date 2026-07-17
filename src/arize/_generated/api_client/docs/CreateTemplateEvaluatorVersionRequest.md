# CreateTemplateEvaluatorVersionRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**commit_message** | **str** | Commit message describing the changes | 
**template_config** | [**TemplateConfig**](TemplateConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_template_evaluator_version_request import CreateTemplateEvaluatorVersionRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateTemplateEvaluatorVersionRequest from a JSON string
create_template_evaluator_version_request_instance = CreateTemplateEvaluatorVersionRequest.from_json(json)
# print the JSON string representation of the object
print(CreateTemplateEvaluatorVersionRequest.to_json())

# convert the object into a dict
create_template_evaluator_version_request_dict = create_template_evaluator_version_request_instance.to_dict()
# create an instance of CreateTemplateEvaluatorVersionRequest from a dict
create_template_evaluator_version_request_from_dict = CreateTemplateEvaluatorVersionRequest.from_dict(create_template_evaluator_version_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


