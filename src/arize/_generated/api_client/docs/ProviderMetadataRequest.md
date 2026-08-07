# ProviderMetadataRequest

Strict request form of ProviderMetadata. For AWS_BEDROCK, must include role_arn. For VERTEX_AI, must include project_id, location, and project_access_label.

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
from arize._generated.api_client.models.provider_metadata_request import ProviderMetadataRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ProviderMetadataRequest from a JSON string
provider_metadata_request_instance = ProviderMetadataRequest.from_json(json)
# print the JSON string representation of the object
print(ProviderMetadataRequest.to_json())

# convert the object into a dict
provider_metadata_request_dict = provider_metadata_request_instance.to_dict()
# create an instance of ProviderMetadataRequest from a dict
provider_metadata_request_from_dict = ProviderMetadataRequest.from_dict(provider_metadata_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


