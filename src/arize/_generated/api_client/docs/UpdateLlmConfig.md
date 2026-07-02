# UpdateLlmConfig

Partial LLM config for PATCH. `provider` is immutable; if present it must match the stored value.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**provider** | [**LlmIntegrationProvider**](LlmIntegrationProvider.md) |  | [optional] 
**api_key** | **str** | Rotate the API key. Pass null to clear it. Omit to keep unchanged. | [optional] 
**is_default_models_enabled** | **bool** |  | [optional] 
**is_function_calling_enabled** | **bool** |  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_llm_config import UpdateLlmConfig

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateLlmConfig from a JSON string
update_llm_config_instance = UpdateLlmConfig.from_json(json)
# print the JSON string representation of the object
print(UpdateLlmConfig.to_json())

# convert the object into a dict
update_llm_config_dict = update_llm_config_instance.to_dict()
# create an instance of UpdateLlmConfig from a dict
update_llm_config_from_dict = UpdateLlmConfig.from_dict(update_llm_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


