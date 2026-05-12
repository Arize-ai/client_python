# LlmGenerationRunConfig

Configuration for running an LLM prompt against each dataset example.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_type** | **str** | Discriminator. Must be &#x60;\&quot;llm_generation\&quot;&#x60;. | 
**ai_integration_id** | **str** | AI integration global ID (base64). | 
**model_name** | **str** | Model name (e.g. &#x60;gpt-4o&#x60;). Falls back to the integration&#39;s default if omitted. | [optional] 
**messages** | [**List[LLMMessage]**](LLMMessage.md) | Array of message objects (at least one). | 
**input_variable_format** | [**InputVariableFormat**](InputVariableFormat.md) |  | 
**invocation_parameters** | [**InvocationParams**](InvocationParams.md) |  | [optional] 
**provider_parameters** | **object** | Provider-specific parameters. Defaults to &#x60;{}&#x60; (no overrides) if omitted. | [optional] 
**tool_config** | [**ToolConfig**](ToolConfig.md) |  | [optional] 
**prompt_version_id** | **str** | Prompt version global ID (base64). Links to a Prompt Hub version for traceability. | [optional] 

## Example

```python
from arize._generated.api_client.models.llm_generation_run_config import LlmGenerationRunConfig

# TODO update the JSON string below
json = "{}"
# create an instance of LlmGenerationRunConfig from a JSON string
llm_generation_run_config_instance = LlmGenerationRunConfig.from_json(json)
# print the JSON string representation of the object
print(LlmGenerationRunConfig.to_json())

# convert the object into a dict
llm_generation_run_config_dict = llm_generation_run_config_instance.to_dict()
# create an instance of LlmGenerationRunConfig from a dict
llm_generation_run_config_from_dict = LlmGenerationRunConfig.from_dict(llm_generation_run_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


