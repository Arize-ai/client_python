# AnthropicConfig

Config for an Anthropic LLM integration.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Whether function/tool calling is enabled. | 
**provider** | **str** | Discriminator identifying the Anthropic provider. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 

## Example

```python
from arize._generated.api_client.models.anthropic_config import AnthropicConfig

# TODO update the JSON string below
json = "{}"
# create an instance of AnthropicConfig from a JSON string
anthropic_config_instance = AnthropicConfig.from_json(json)
# print the JSON string representation of the object
print(AnthropicConfig.to_json())

# convert the object into a dict
anthropic_config_dict = anthropic_config_instance.to_dict()
# create an instance of AnthropicConfig from a dict
anthropic_config_from_dict = AnthropicConfig.from_dict(anthropic_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


