# CreateAnthropicConfig

Create config for an Anthropic LLM integration. `api_key` is required and is write-only (never returned in responses).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Enable function/tool calling. Defaults to true. | [optional] 
**provider** | **str** |  | 
**api_key** | **str** | API key for the provider (write-only, never returned). | 

## Example

```python
from arize._generated.api_client.models.create_anthropic_config import CreateAnthropicConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAnthropicConfig from a JSON string
create_anthropic_config_instance = CreateAnthropicConfig.from_json(json)
# print the JSON string representation of the object
print(CreateAnthropicConfig.to_json())

# convert the object into a dict
create_anthropic_config_dict = create_anthropic_config_instance.to_dict()
# create an instance of CreateAnthropicConfig from a dict
create_anthropic_config_from_dict = CreateAnthropicConfig.from_dict(create_anthropic_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


