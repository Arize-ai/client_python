# LiteLlmConfig

Config for a LiteLLM integration. `base_url` is the LiteLLM endpoint Arize sends requests to and is always set — LiteLLM is self-hosted, so there is no default endpoint. Secrets are write-only: the virtual key surfaces as `has_api_key` and custom request headers surface as `header_names` (names only). `model_names` lists only the model names configured on this integration; models resolved live from the LiteLLM deployment are served through the Arize UI and are not returned here.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Whether function/tool calling is enabled. | 
**provider** | **str** | Discriminator identifying the LiteLLM provider. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 
**base_url** | **str** | LiteLLM endpoint URL requests are sent to. | 
**header_names** | **List[str]** | Names of the custom request headers configured on this integration. Empty when none are configured. Header values are write-only and never returned. | 
**model_names** | **List[str]** | Custom model names configured on this integration. Empty when none. | 

## Example

```python
from arize._generated.api_client.models.lite_llm_config import LiteLlmConfig

# TODO update the JSON string below
json = "{}"
# create an instance of LiteLlmConfig from a JSON string
lite_llm_config_instance = LiteLlmConfig.from_json(json)
# print the JSON string representation of the object
print(LiteLlmConfig.to_json())

# convert the object into a dict
lite_llm_config_dict = lite_llm_config_instance.to_dict()
# create an instance of LiteLlmConfig from a dict
lite_llm_config_from_dict = LiteLlmConfig.from_dict(lite_llm_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


