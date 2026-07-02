# CreateOpenAiConfig

Create config for an OpenAI LLM integration. `api_key` is required and is write-only (never returned in responses).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_default_models_enabled** | **bool** | Enable the provider&#39;s default model list. Defaults to false. | [optional] 
**is_function_calling_enabled** | **bool** | Enable function/tool calling. Defaults to true. | [optional] 
**provider** | **str** |  | 
**api_key** | **str** | API key for the provider (write-only, never returned). | 

## Example

```python
from arize._generated.api_client.models.create_open_ai_config import CreateOpenAiConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateOpenAiConfig from a JSON string
create_open_ai_config_instance = CreateOpenAiConfig.from_json(json)
# print the JSON string representation of the object
print(CreateOpenAiConfig.to_json())

# convert the object into a dict
create_open_ai_config_dict = create_open_ai_config_instance.to_dict()
# create an instance of CreateOpenAiConfig from a dict
create_open_ai_config_from_dict = CreateOpenAiConfig.from_dict(create_open_ai_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


