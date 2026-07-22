# LlmConfig

Per-provider LLM config, discriminated by `provider`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Whether function/tool calling is enabled. | 
**provider** | **str** | Discriminator identifying the NVIDIA NIM provider. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 
**is_default_models_enabled** | **bool** | Whether Arize&#39;s default model catalog is enabled. | 
**model_names** | **List[str]** | Custom model names configured on this integration. Empty when none. | 
**auth** | [**AwsBedrockAuth**](AwsBedrockAuth.md) |  | 
**base_url** | **str** | Self-hosted NIM endpoint URL. Null when not set. | 
**header_names** | **List[str]** | Names of the custom request headers configured on this integration. Empty when none are configured. Header values are write-only and never returned. | 
**project_id** | **str** | GCP project ID Arize accesses Vertex through. | 
**location** | **str** | GCP region (e.g. us-central1). | 
**project_access_label** | **str** | Label used to verify Arize&#39;s access to the GCP project. | 

## Example

```python
from arize._generated.api_client.models.llm_config import LlmConfig

# TODO update the JSON string below
json = "{}"
# create an instance of LlmConfig from a JSON string
llm_config_instance = LlmConfig.from_json(json)
# print the JSON string representation of the object
print(LlmConfig.to_json())

# convert the object into a dict
llm_config_dict = llm_config_instance.to_dict()
# create an instance of LlmConfig from a dict
llm_config_from_dict = LlmConfig.from_dict(llm_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


