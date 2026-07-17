# CreateCodeEvaluatorVersionRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**commit_message** | **str** | Commit message describing the changes | 
**code_config** | [**CodeConfig**](CodeConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_code_evaluator_version_request import CreateCodeEvaluatorVersionRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateCodeEvaluatorVersionRequest from a JSON string
create_code_evaluator_version_request_instance = CreateCodeEvaluatorVersionRequest.from_json(json)
# print the JSON string representation of the object
print(CreateCodeEvaluatorVersionRequest.to_json())

# convert the object into a dict
create_code_evaluator_version_request_dict = create_code_evaluator_version_request_instance.to_dict()
# create an instance of CreateCodeEvaluatorVersionRequest from a dict
create_code_evaluator_version_request_from_dict = CreateCodeEvaluatorVersionRequest.from_dict(create_code_evaluator_version_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


