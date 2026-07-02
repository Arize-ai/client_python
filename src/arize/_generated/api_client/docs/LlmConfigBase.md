# LlmConfigBase

LLM config fields common across providers.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_default_models_enabled** | **bool** | Whether the provider&#39;s default model list is enabled. | 
**is_function_calling_enabled** | **bool** | Whether function/tool calling is enabled. | 

## Example

```python
from arize._generated.api_client.models.llm_config_base import LlmConfigBase

# TODO update the JSON string below
json = "{}"
# create an instance of LlmConfigBase from a JSON string
llm_config_base_instance = LlmConfigBase.from_json(json)
# print the JSON string representation of the object
print(LlmConfigBase.to_json())

# convert the object into a dict
llm_config_base_dict = llm_config_base_instance.to_dict()
# create an instance of LlmConfigBase from a dict
llm_config_base_from_dict = LlmConfigBase.from_dict(llm_config_base_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


