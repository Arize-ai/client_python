# CreateDeepSeekConfig

Create config for a DeepSeek integration. `api_key` is required and is write-only (never returned; it surfaces as `has_api_key` on read). DeepSeek is a single hosted service, so there is no endpoint field and no custom request headers. Neither `is_default_models_enabled` nor `model_names` is required: Arize resolves the models the key can reach from the DeepSeek account, so an integration created with neither still has a selectable model list.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Enable function/tool calling. Defaults to true. | [optional] 
**provider** | **str** | Discriminator identifying the DeepSeek provider. | 
**api_key** | **str** | DeepSeek API key (write-only, never returned). | 
**is_default_models_enabled** | **bool** | Enable Arize&#39;s default model catalog. Defaults to false. | [optional] 
**model_names** | **List[str]** | Custom model names to make available. Defaults to an empty list. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_deep_seek_config import CreateDeepSeekConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateDeepSeekConfig from a JSON string
create_deep_seek_config_instance = CreateDeepSeekConfig.from_json(json)
# print the JSON string representation of the object
print(CreateDeepSeekConfig.to_json())

# convert the object into a dict
create_deep_seek_config_dict = create_deep_seek_config_instance.to_dict()
# create an instance of CreateDeepSeekConfig from a dict
create_deep_seek_config_from_dict = CreateDeepSeekConfig.from_dict(create_deep_seek_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


