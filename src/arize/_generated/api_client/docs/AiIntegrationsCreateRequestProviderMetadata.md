# AiIntegrationsCreateRequestProviderMetadata

Provider-specific configuration. For awsBedrock, must include role_arn. For vertexAI, must include project_id, location, and project_access_label.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**kind** | [**GcpProviderMetadataKind**](GcpProviderMetadataKind.md) |  | 
**role_arn** | **str** | AWS IAM role ARN for cross-account access | 
**external_id** | **str** | External ID for the assume-role policy | [optional] 
**project_id** | **str** | GCP project ID | 
**location** | **str** | GCP region (e.g. us-central1) | 
**project_access_label** | **str** | Display label for the project | 

## Example

```python
from arize._generated.api_client.models.ai_integrations_create_request_provider_metadata import AiIntegrationsCreateRequestProviderMetadata

# TODO update the JSON string below
json = "{}"
# create an instance of AiIntegrationsCreateRequestProviderMetadata from a JSON string
ai_integrations_create_request_provider_metadata_instance = AiIntegrationsCreateRequestProviderMetadata.from_json(json)
# print the JSON string representation of the object
print(AiIntegrationsCreateRequestProviderMetadata.to_json())

# convert the object into a dict
ai_integrations_create_request_provider_metadata_dict = ai_integrations_create_request_provider_metadata_instance.to_dict()
# create an instance of AiIntegrationsCreateRequestProviderMetadata from a dict
ai_integrations_create_request_provider_metadata_from_dict = AiIntegrationsCreateRequestProviderMetadata.from_dict(ai_integrations_create_request_provider_metadata_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


