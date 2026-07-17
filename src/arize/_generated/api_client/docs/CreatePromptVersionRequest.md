# CreatePromptVersionRequest

Prompt version creation parameters.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**commit_message** | **str** | Commit message describing this version | 
**input_variable_format** | [**InputVariableFormat**](InputVariableFormat.md) |  | [optional] 
**provider** | [**LlmProvider**](LlmProvider.md) |  | 
**model** | **str** | The model to use for the call. Optional. If omitted, no default model is set on the version. | [optional] 
**messages** | [**List[LLMMessage]**](LLMMessage.md) | The messages that make up the prompt template | 
**invocation_params** | [**InvocationParams**](InvocationParams.md) |  | [optional] 
**provider_params** | [**ProviderParams**](ProviderParams.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.create_prompt_version_request import CreatePromptVersionRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreatePromptVersionRequest from a JSON string
create_prompt_version_request_instance = CreatePromptVersionRequest.from_json(json)
# print the JSON string representation of the object
print(CreatePromptVersionRequest.to_json())

# convert the object into a dict
create_prompt_version_request_dict = create_prompt_version_request_instance.to_dict()
# create an instance of CreatePromptVersionRequest from a dict
create_prompt_version_request_from_dict = CreatePromptVersionRequest.from_dict(create_prompt_version_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


