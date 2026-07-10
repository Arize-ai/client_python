# OpenAiConfig

Config for an OpenAI LLM integration.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Whether function/tool calling is enabled. | 
**provider** | **str** | Discriminator identifying the OpenAI provider. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 

## Example

```python
from arize._generated.api_client.models.open_ai_config import OpenAiConfig

# TODO update the JSON string below
json = "{}"
# create an instance of OpenAiConfig from a JSON string
open_ai_config_instance = OpenAiConfig.from_json(json)
# print the JSON string representation of the object
print(OpenAiConfig.to_json())

# convert the object into a dict
open_ai_config_dict = open_ai_config_instance.to_dict()
# create an instance of OpenAiConfig from a dict
open_ai_config_from_dict = OpenAiConfig.from_dict(open_ai_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


