# TogetherAiConfig

Config for a Together AI integration. Together AI is a single hosted service, so there is no endpoint field and no custom request headers. The API key is write-only and surfaces as `has_api_key`. `model_names` lists only the model names configured on this integration; models resolved live from the Together AI account are served through the Arize UI and are not returned here.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Whether function/tool calling is enabled. | 
**provider** | **str** | Discriminator identifying the Together AI provider. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 
**is_default_models_enabled** | **bool** | Whether Arize&#39;s default model catalog is enabled. | 
**model_names** | **List[str]** | Custom model names configured on this integration. Empty when none. | 

## Example

```python
from arize._generated.api_client.models.together_ai_config import TogetherAiConfig

# TODO update the JSON string below
json = "{}"
# create an instance of TogetherAiConfig from a JSON string
together_ai_config_instance = TogetherAiConfig.from_json(json)
# print the JSON string representation of the object
print(TogetherAiConfig.to_json())

# convert the object into a dict
together_ai_config_dict = together_ai_config_instance.to_dict()
# create an instance of TogetherAiConfig from a dict
together_ai_config_from_dict = TogetherAiConfig.from_dict(together_ai_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


