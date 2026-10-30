# CloudflareProviderMetadata

Cloudflare provider metadata

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**kind** | [**CloudflareProviderMetadataKind**](CloudflareProviderMetadataKind.md) |  | 
**account_id** | **str** | The 32-character Cloudflare account ID | 

## Example

```python
from arize._generated.api_client.models.cloudflare_provider_metadata import CloudflareProviderMetadata

# TODO update the JSON string below
json = "{}"
# create an instance of CloudflareProviderMetadata from a JSON string
cloudflare_provider_metadata_instance = CloudflareProviderMetadata.from_json(json)
# print the JSON string representation of the object
print(CloudflareProviderMetadata.to_json())

# convert the object into a dict
cloudflare_provider_metadata_dict = cloudflare_provider_metadata_instance.to_dict()
# create an instance of CloudflareProviderMetadata from a dict
cloudflare_provider_metadata_from_dict = CloudflareProviderMetadata.from_dict(cloudflare_provider_metadata_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


