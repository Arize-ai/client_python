# FireworksConfig

Config for a Fireworks AI integration. Fireworks is a single hosted service, so there is no endpoint field and no custom request headers. The API key is write-only and surfaces as `has_api_key`. `model_names` lists only the model names configured on this integration; models resolved live from the Fireworks account are served through the Arize UI and are not returned here.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Whether function/tool calling is enabled. | 
**provider** | **str** | Discriminator identifying the Fireworks AI provider. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 
**is_default_models_enabled** | **bool** | Whether Arize&#39;s default model catalog is enabled. | 
**model_names** | **List[str]** | Custom model names configured on this integration. Empty when none. | 

## Example

```python
from arize._generated.api_client.models.fireworks_config import FireworksConfig

# TODO update the JSON string below
json = "{}"
# create an instance of FireworksConfig from a JSON string
fireworks_config_instance = FireworksConfig.from_json(json)
# print the JSON string representation of the object
print(FireworksConfig.to_json())

# convert the object into a dict
fireworks_config_dict = fireworks_config_instance.to_dict()
# create an instance of FireworksConfig from a dict
fireworks_config_from_dict = FireworksConfig.from_dict(fireworks_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


