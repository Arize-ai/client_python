# PromptVersionCreateRequest

Initial version configuration for a new prompt

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**commit_message** | **str** | Commit message describing this version | 
**input_variable_format** | [**InputVariableFormat**](InputVariableFormat.md) |  | 
**provider** | [**LlmProvider**](LlmProvider.md) |  | 
**model** | **str** | The model to use for the call. Optional. If omitted, no default model is set on the prompt version. | [optional] 
**messages** | [**List[LLMMessage]**](LLMMessage.md) | The messages that make up the prompt template | 
**invocation_params** | [**InvocationParams**](InvocationParams.md) |  | [optional] 
**provider_params** | [**ProviderParams**](ProviderParams.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.prompt_version_create_request import PromptVersionCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of PromptVersionCreateRequest from a JSON string
prompt_version_create_request_instance = PromptVersionCreateRequest.from_json(json)
# print the JSON string representation of the object
print(PromptVersionCreateRequest.to_json())

# convert the object into a dict
prompt_version_create_request_dict = prompt_version_create_request_instance.to_dict()
# create an instance of PromptVersionCreateRequest from a dict
prompt_version_create_request_from_dict = PromptVersionCreateRequest.from_dict(prompt_version_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


