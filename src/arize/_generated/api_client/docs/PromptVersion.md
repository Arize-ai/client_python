# PromptVersion

A prompt version represents a specific snapshot of a prompt's configuration. Each version captures the messages, model settings, and parameters at a point in time. Versions are immutable once created and are identified by a commit hash. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The prompt version ID | 
**prompt_id** | **str** | The prompt ID this version belongs to | 
**commit_hash** | **str** | The commit hash of this version | 
**commit_message** | **str** | The commit message describing the changes in this version | 
**messages** | [**List[LLMMessage]**](LLMMessage.md) | The messages that make up the prompt template | 
**input_variable_format** | [**InputVariableFormat**](InputVariableFormat.md) |  | 
**provider** | [**LlmProvider**](LlmProvider.md) |  | 
**model** | **str** | The model to use for the call | 
**invocation_params** | [**InvocationParams**](InvocationParams.md) |  | [optional] 
**provider_params** | [**ProviderParams**](ProviderParams.md) |  | [optional] 
**tool_config** | [**ToolConfig**](ToolConfig.md) |  | [optional] 
**created_at** | **datetime** | When the version was created | 
**created_by_user_id** | **str** | The user ID of the user who created this version | 
**labels** | **List[str]** | Label names currently pointing to this version (e.g., \&quot;production\&quot;, \&quot;staging\&quot;) | [optional] 

## Example

```python
from arize._generated.api_client.models.prompt_version import PromptVersion

# TODO update the JSON string below
json = "{}"
# create an instance of PromptVersion from a JSON string
prompt_version_instance = PromptVersion.from_json(json)
# print the JSON string representation of the object
print(PromptVersion.to_json())

# convert the object into a dict
prompt_version_dict = prompt_version_instance.to_dict()
# create an instance of PromptVersion from a dict
prompt_version_from_dict = PromptVersion.from_dict(prompt_version_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


