# CreateLlmConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**is_function_calling_enabled** | **bool** | Enable function/tool calling. Defaults to true. | [optional] 
**provider** | **str** |  | 
**api_key** | **str** | API key for the endpoint (write-only, never returned). | 
**auth** | [**CreateAwsBedrockAuth**](CreateAwsBedrockAuth.md) |  | 
**is_default_models_enabled** | **bool** | Enable Arize&#39;s default model catalog. Defaults to false. | [optional] 
**model_names** | **List[str]** | Custom model names to make available. Defaults to none. | [optional] 
**base_url** | **str** | Self-hosted NIM endpoint URL (HTTPS). Defaults to the provider default endpoint. | 
**headers** | **Dict[str, str]** | Custom request headers sent to the endpoint, as a name-to-value map. Write-only: values are never returned; names are exposed as &#x60;header_names&#x60; on read. Defaults to no headers. The serialized header map must not exceed 8,175 bytes. | [optional] 
**project_id** | **str** | GCP project ID Arize accesses Vertex through. | 
**location** | **str** | GCP region (e.g. us-central1). | 
**project_access_label** | **str** | Label used to verify Arize&#39;s access to the GCP project. | 

## Example

```python
from arize._generated.api_client.models.create_llm_config import CreateLlmConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateLlmConfig from a JSON string
create_llm_config_instance = CreateLlmConfig.from_json(json)
# print the JSON string representation of the object
print(CreateLlmConfig.to_json())

# convert the object into a dict
create_llm_config_dict = create_llm_config_instance.to_dict()
# create an instance of CreateLlmConfig from a dict
create_llm_config_from_dict = CreateLlmConfig.from_dict(create_llm_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


