# UpdateAiIntegrationRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New integration name | [optional] 
**provider** | [**AiIntegrationProvider**](AiIntegrationProvider.md) |  | [optional] 
**api_key** | **str** | New API key. Pass null to remove the existing key. Omit to keep unchanged. | [optional] 
**base_url** | **str** | Custom base URL. Pass null to remove. | [optional] 
**model_names** | **List[str]** | Supported model names (replaces all) | [optional] 
**headers** | **Dict[str, str]** | Custom headers. Pass null to remove. The serialized header map must not exceed 8,175 bytes. | [optional] 
**enable_default_models** | **bool** | Enable provider&#39;s default model list | [optional] 
**function_calling_enabled** | **bool** | Enable function/tool calling | [optional] 
**auth_type** | [**AiIntegrationAuthType**](AiIntegrationAuthType.md) |  | [optional] 
**provider_metadata** | [**ProviderMetadata**](ProviderMetadata.md) | Provider-specific configuration. For AWS_BEDROCK, must include role_arn. For VERTEX_AI, must include project_id, location, and project_access_label. Pass null to remove. | [optional] 
**scopings** | [**List[AiIntegrationScoping]**](AiIntegrationScoping.md) | Visibility scoping rules (replaces all existing scopings) | [optional] 

## Example

```python
from arize._generated.api_client.models.update_ai_integration_request import UpdateAiIntegrationRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateAiIntegrationRequest from a JSON string
update_ai_integration_request_instance = UpdateAiIntegrationRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateAiIntegrationRequest.to_json())

# convert the object into a dict
update_ai_integration_request_dict = update_ai_integration_request_instance.to_dict()
# create an instance of UpdateAiIntegrationRequest from a dict
update_ai_integration_request_from_dict = UpdateAiIntegrationRequest.from_dict(update_ai_integration_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


