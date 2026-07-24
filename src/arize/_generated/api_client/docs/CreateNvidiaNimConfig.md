# CreateNvidiaNimConfig

Create config for an NVIDIA NIM integration. Every connection field is optional: omit `base_url` to use the provider default endpoint, or set it to a self-hosted NIM endpoint (validated server-side). `api_key` and `headers` are write-only (never returned; headers surface as `header_names` on read). The integration must have at least one model source: enable `is_default_models_enabled` or provide at least one entry in `model_names`, otherwise the request is rejected with 422.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Enable function/tool calling. Defaults to true. | [optional] 
**provider** | **str** |  | 
**base_url** | **str** | Self-hosted NIM endpoint URL (HTTPS). Defaults to the provider default endpoint. | [optional] 
**api_key** | **str** | API key for the endpoint (write-only, never returned). | [optional] 
**headers** | **Dict[str, str]** | Custom request headers sent to the endpoint, as a name-to-value map. Write-only: values are never returned; names are exposed as &#x60;header_names&#x60; on read. Defaults to no headers. The serialized header map must not exceed 8,175 bytes. | [optional] 
**is_default_models_enabled** | **bool** | Enable Arize&#39;s default model catalog. Defaults to false. | [optional] 
**model_names** | **List[str]** | Custom model names to make available. Defaults to none. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_nvidia_nim_config import CreateNvidiaNimConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateNvidiaNimConfig from a JSON string
create_nvidia_nim_config_instance = CreateNvidiaNimConfig.from_json(json)
# print the JSON string representation of the object
print(CreateNvidiaNimConfig.to_json())

# convert the object into a dict
create_nvidia_nim_config_dict = create_nvidia_nim_config_instance.to_dict()
# create an instance of CreateNvidiaNimConfig from a dict
create_nvidia_nim_config_from_dict = CreateNvidiaNimConfig.from_dict(create_nvidia_nim_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


