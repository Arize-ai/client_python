# GcpProviderMetadata

Vertex AI (GCP) provider metadata

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**kind** | [**GcpProviderMetadataKind**](GcpProviderMetadataKind.md) |  | 
**project_id** | **str** | GCP project ID | 
**location** | **str** | GCP region (e.g. us-central1) | 
**project_access_label** | **str** | Display label for the project | 

## Example

```python
from arize._generated.api_client.models.gcp_provider_metadata import GcpProviderMetadata

# TODO update the JSON string below
json = "{}"
# create an instance of GcpProviderMetadata from a JSON string
gcp_provider_metadata_instance = GcpProviderMetadata.from_json(json)
# print the JSON string representation of the object
print(GcpProviderMetadata.to_json())

# convert the object into a dict
gcp_provider_metadata_dict = gcp_provider_metadata_instance.to_dict()
# create an instance of GcpProviderMetadata from a dict
gcp_provider_metadata_from_dict = GcpProviderMetadata.from_dict(gcp_provider_metadata_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


