# AiIntegration

An AI integration configures access to an external LLM provider (e.g. OpenAI, Azure OpenAI, AWS Bedrock, Vertex AI). Integrations can be scoped to the entire account, a specific organization, or a specific space. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The integration ID | 
**name** | **str** | The integration name | 
**provider** | [**AiIntegrationProvider**](AiIntegrationProvider.md) |  | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned) | 
**base_url** | **str** | Custom base URL for the provider | [optional] 
**model_names** | **List[str]** | Supported model names | [optional] 
**headers** | **Dict[str, str]** | Custom headers included in requests | [optional] 
**enable_default_models** | **bool** | Whether the provider&#39;s default model list is enabled | 
**function_calling_enabled** | **bool** | Whether function/tool calling is enabled | 
**auth_type** | [**AiIntegrationAuthType**](AiIntegrationAuthType.md) |  | 
**provider_metadata** | **Dict[str, object]** | Provider-specific configuration (AWS or GCP metadata) | [optional] 
**scopings** | [**List[AiIntegrationScoping]**](AiIntegrationScoping.md) | Visibility scoping rules | 
**created_at** | **datetime** | When the integration was created | 
**updated_at** | **datetime** | When the integration was last updated | 
**created_by_user_id** | **str** | The user ID of the user who created the integration | 

## Example

```python
from arize._generated.api_client.models.ai_integration import AiIntegration

# TODO update the JSON string below
json = "{}"
# create an instance of AiIntegration from a JSON string
ai_integration_instance = AiIntegration.from_json(json)
# print the JSON string representation of the object
print(AiIntegration.to_json())

# convert the object into a dict
ai_integration_dict = ai_integration_instance.to_dict()
# create an instance of AiIntegration from a dict
ai_integration_from_dict = AiIntegration.from_dict(ai_integration_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


