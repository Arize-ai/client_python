# GcpProviderMetadataRequest

Vertex AI (GCP) provider metadata in a write request (strict form of GcpProviderMetadata).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**kind** | [**GcpProviderMetadataKind**](GcpProviderMetadataKind.md) |  | 
**project_id** | **str** | GCP project ID | 
**location** | **str** | GCP region (e.g. us-central1) | 
**project_access_label** | **str** | Display label for the project | 

## Example

```python
from arize._generated.api_client.models.gcp_provider_metadata_request import GcpProviderMetadataRequest

# TODO update the JSON string below
json = "{}"
# create an instance of GcpProviderMetadataRequest from a JSON string
gcp_provider_metadata_request_instance = GcpProviderMetadataRequest.from_json(json)
# print the JSON string representation of the object
print(GcpProviderMetadataRequest.to_json())

# convert the object into a dict
gcp_provider_metadata_request_dict = gcp_provider_metadata_request_instance.to_dict()
# create an instance of GcpProviderMetadataRequest from a dict
gcp_provider_metadata_request_from_dict = GcpProviderMetadataRequest.from_dict(gcp_provider_metadata_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


