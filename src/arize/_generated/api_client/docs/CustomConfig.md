# CustomConfig

Config for a custom OpenAI-compatible endpoint integration. `base_url` is the endpoint Arize sends requests to; it must implement the OpenAI API shape. Secrets are write-only: the API key surfaces as `has_api_key` and custom request headers surface as `header_names` (names only).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Whether function/tool calling is enabled. | 
**provider** | **str** | Discriminator identifying a custom OpenAI-compatible endpoint. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 
**base_url** | **str** | Endpoint URL requests are sent to. | 
**header_names** | **List[str]** | Names of the custom request headers configured on this integration. Empty when none are configured. Header values are write-only and never returned. | 
**is_default_models_enabled** | **bool** | Whether Arize&#39;s default model catalog is enabled. | 
**model_names** | **List[str]** | Custom model names configured on this integration. Empty when none. | 

## Example

```python
from arize._generated.api_client.models.custom_config import CustomConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CustomConfig from a JSON string
custom_config_instance = CustomConfig.from_json(json)
# print the JSON string representation of the object
print(CustomConfig.to_json())

# convert the object into a dict
custom_config_dict = custom_config_instance.to_dict()
# create an instance of CustomConfig from a dict
custom_config_from_dict = CustomConfig.from_dict(custom_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


