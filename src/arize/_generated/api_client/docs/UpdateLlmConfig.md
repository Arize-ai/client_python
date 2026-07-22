# UpdateLlmConfig

Partial LLM config for PATCH. `provider` is immutable; if present it must match the stored value. Field applicability is provider-specific and enforced by the handler with 422: `api_key` and `is_function_calling_enabled` do not apply to `AWS_BEDROCK` or `VERTEX_AI`; `auth` applies to `AWS_BEDROCK` only; `base_url` and `headers` apply to `CUSTOM` and `NVIDIA_NIM` only; `is_default_models_enabled` and `model_names` apply to `AWS_BEDROCK`, `CUSTOM`, and `NVIDIA_NIM` only; `project_id`, `location`, and `project_access_label` apply to `VERTEX_AI` only.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**provider** | [**LlmIntegrationProvider**](LlmIntegrationProvider.md) |  | [optional] 
**api_key** | **str** | Rotate the API key. Pass null to clear it. Omit to keep unchanged. Not valid for &#x60;AWS_BEDROCK&#x60; (bearer tokens are rotated via &#x60;auth&#x60;). | [optional] 
**is_function_calling_enabled** | **bool** | Enable or disable function/tool calling. Omit to keep unchanged. Not valid for &#x60;AWS_BEDROCK&#x60;. | [optional] 
**auth** | [**CreateAwsBedrockAuth**](CreateAwsBedrockAuth.md) |  | [optional] 
**base_url** | **str** | (&#x60;CUSTOM&#x60; and &#x60;NVIDIA_NIM&#x60; only) New endpoint URL. For &#x60;NVIDIA_NIM&#x60; the field is optional on the resource, so null clears it (falling back to the provider default endpoint). For &#x60;CUSTOM&#x60; it is required on the resource — null is rejected with 422. Omit to keep unchanged. | [optional] 
**headers** | **Dict[str, str]** | (&#x60;CUSTOM&#x60; and &#x60;NVIDIA_NIM&#x60; only) Replaces the configured custom request headers: the provided map becomes the full header set. Pass null to clear all headers. Omit to keep unchanged. Write-only; names are exposed as &#x60;header_names&#x60; on read. | [optional] 
**is_default_models_enabled** | **bool** | (&#x60;AWS_BEDROCK&#x60;, &#x60;CUSTOM&#x60;, and &#x60;NVIDIA_NIM&#x60; only) Enable or disable Arize&#39;s default model catalog. The effective config must keep at least one model source or the request is rejected with 422. Omit to keep unchanged. | [optional] 
**model_names** | **List[str]** | (&#x60;AWS_BEDROCK&#x60;, &#x60;CUSTOM&#x60;, and &#x60;NVIDIA_NIM&#x60; only) Replaces the custom model list. The effective config must keep at least one model source or the request is rejected with 422. Omit to keep unchanged. | [optional] 
**project_id** | **str** | (&#x60;VERTEX_AI&#x60; only) New GCP project ID. Required on the resource, so it may be changed but never cleared; omitted fields keep their stored values (per-scalar deep-merge). | [optional] 
**location** | **str** | (&#x60;VERTEX_AI&#x60; only) New GCP region. Required on the resource, so it may be changed but never cleared; omitted fields keep their stored values (per-scalar deep-merge). | [optional] 
**project_access_label** | **str** | (&#x60;VERTEX_AI&#x60; only) New project-access label. Required on the resource, so it may be changed but never cleared; omitted fields keep their stored values (per-scalar deep-merge). | [optional] 

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


