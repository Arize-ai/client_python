# CloudflareProviderMetadataRequest

Cloudflare provider metadata in a write request (strict form of CloudflareProviderMetadata).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**kind** | [**CloudflareProviderMetadataKind**](CloudflareProviderMetadataKind.md) |  | 
**account_id** | **str** | The 32-character Cloudflare account ID | 

## Example

```python
from arize._generated.api_client.models.cloudflare_provider_metadata_request import CloudflareProviderMetadataRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CloudflareProviderMetadataRequest from a JSON string
cloudflare_provider_metadata_request_instance = CloudflareProviderMetadataRequest.from_json(json)
# print the JSON string representation of the object
print(CloudflareProviderMetadataRequest.to_json())

# convert the object into a dict
cloudflare_provider_metadata_request_dict = cloudflare_provider_metadata_request_instance.to_dict()
# create an instance of CloudflareProviderMetadataRequest from a dict
cloudflare_provider_metadata_request_from_dict = CloudflareProviderMetadataRequest.from_dict(cloudflare_provider_metadata_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


