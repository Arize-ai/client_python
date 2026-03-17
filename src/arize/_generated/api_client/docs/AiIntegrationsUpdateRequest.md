# AiIntegrationsUpdateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New integration name | [optional] 
**provider** | [**AiIntegrationProvider**](AiIntegrationProvider.md) |  | [optional] 
**api_key** | **str** | New API key. Pass null to remove the existing key. Omit to keep unchanged. | [optional] 
**base_url** | **str** | Custom base URL. Pass null to remove. | [optional] 
**model_names** | **List[str]** | Supported model names (replaces all) | [optional] 
**headers** | **Dict[str, str]** | Custom headers. Pass null to remove. | [optional] 
**enable_default_models** | **bool** | Enable provider&#39;s default model list | [optional] 
**function_calling_enabled** | **bool** | Enable function/tool calling | [optional] 
**auth_type** | [**AiIntegrationAuthType**](AiIntegrationAuthType.md) |  | [optional] 
**provider_metadata** | **Dict[str, object]** | Provider-specific configuration | [optional] 
**scopings** | [**List[AiIntegrationScoping]**](AiIntegrationScoping.md) | Visibility scoping rules (replaces all existing scopings) | [optional] 

## Example

```python
from arize._generated.api_client.models.ai_integrations_update_request import AiIntegrationsUpdateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AiIntegrationsUpdateRequest from a JSON string
ai_integrations_update_request_instance = AiIntegrationsUpdateRequest.from_json(json)
# print the JSON string representation of the object
print(AiIntegrationsUpdateRequest.to_json())

# convert the object into a dict
ai_integrations_update_request_dict = ai_integrations_update_request_instance.to_dict()
# create an instance of AiIntegrationsUpdateRequest from a dict
ai_integrations_update_request_from_dict = AiIntegrationsUpdateRequest.from_dict(ai_integrations_update_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


