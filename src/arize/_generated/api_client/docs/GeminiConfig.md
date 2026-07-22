# GeminiConfig

Config for a Google Gemini LLM integration.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Whether function/tool calling is enabled. | 
**provider** | **str** | Discriminator identifying the Gemini provider. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 

## Example

```python
from arize._generated.api_client.models.gemini_config import GeminiConfig

# TODO update the JSON string below
json = "{}"
# create an instance of GeminiConfig from a JSON string
gemini_config_instance = GeminiConfig.from_json(json)
# print the JSON string representation of the object
print(GeminiConfig.to_json())

# convert the object into a dict
gemini_config_dict = gemini_config_instance.to_dict()
# create an instance of GeminiConfig from a dict
gemini_config_from_dict = GeminiConfig.from_dict(gemini_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


