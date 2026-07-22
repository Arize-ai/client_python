# NvidiaNimConfig

Config for an NVIDIA NIM integration. Every connection field is optional: `base_url` targets a self-hosted NIM endpoint (null when using the provider default). Secrets are write-only: the API key surfaces as `has_api_key` and custom request headers surface as `header_names` (names only).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Whether function/tool calling is enabled. | 
**provider** | **str** | Discriminator identifying the NVIDIA NIM provider. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 
**base_url** | **str** | Self-hosted NIM endpoint URL. Null when not set. | 
**header_names** | **List[str]** | Names of the custom request headers configured on this integration. Empty when none are configured. Header values are write-only and never returned. | 
**is_default_models_enabled** | **bool** | Whether Arize&#39;s default model catalog is enabled. | 
**model_names** | **List[str]** | Custom model names configured on this integration. Empty when none. | 

## Example

```python
from arize._generated.api_client.models.nvidia_nim_config import NvidiaNimConfig

# TODO update the JSON string below
json = "{}"
# create an instance of NvidiaNimConfig from a JSON string
nvidia_nim_config_instance = NvidiaNimConfig.from_json(json)
# print the JSON string representation of the object
print(NvidiaNimConfig.to_json())

# convert the object into a dict
nvidia_nim_config_dict = nvidia_nim_config_instance.to_dict()
# create an instance of NvidiaNimConfig from a dict
nvidia_nim_config_from_dict = NvidiaNimConfig.from_dict(nvidia_nim_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


