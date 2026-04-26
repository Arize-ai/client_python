# AiIntegrationsCreateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Integration name | 
**provider** | [**AiIntegrationProvider**](AiIntegrationProvider.md) |  | 
**api_key** | **str** | API key for the provider (write-only, never returned) | [optional] 
**base_url** | **str** | Custom base URL for the provider | [optional] 
**model_names** | **List[str]** | Supported model names | [optional] 
**headers** | **Dict[str, str]** | Custom headers to include in requests | [optional] 
**enable_default_models** | **bool** | Enable provider&#39;s default model list (default false) | [optional] 
**function_calling_enabled** | **bool** | Enable function/tool calling (default true) | [optional] 
**auth_type** | [**AiIntegrationAuthType**](AiIntegrationAuthType.md) |  | [optional] 
**provider_metadata** | [**AiIntegrationsCreateRequestProviderMetadata**](AiIntegrationsCreateRequestProviderMetadata.md) |  | [optional] 
**scopings** | [**List[AiIntegrationScoping]**](AiIntegrationScoping.md) | Visibility scoping rules. Defaults to account-wide. | [optional] 

## Example

```python
from arize._generated.api_client.models.ai_integrations_create_request import AiIntegrationsCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AiIntegrationsCreateRequest from a JSON string
ai_integrations_create_request_instance = AiIntegrationsCreateRequest.from_json(json)
# print the JSON string representation of the object
print(AiIntegrationsCreateRequest.to_json())

# convert the object into a dict
ai_integrations_create_request_dict = ai_integrations_create_request_instance.to_dict()
# create an instance of AiIntegrationsCreateRequest from a dict
ai_integrations_create_request_from_dict = AiIntegrationsCreateRequest.from_dict(ai_integrations_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


