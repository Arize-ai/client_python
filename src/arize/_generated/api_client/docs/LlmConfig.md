# LlmConfig

Per-provider LLM config, discriminated by `provider`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Whether function/tool calling is enabled. | 
**provider** | **str** | Discriminator identifying the Anthropic provider. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 

## Example

```python
from arize._generated.api_client.models.llm_config import LlmConfig

# TODO update the JSON string below
json = "{}"
# create an instance of LlmConfig from a JSON string
llm_config_instance = LlmConfig.from_json(json)
# print the JSON string representation of the object
print(LlmConfig.to_json())

# convert the object into a dict
llm_config_dict = llm_config_instance.to_dict()
# create an instance of LlmConfig from a dict
llm_config_from_dict = LlmConfig.from_dict(llm_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


