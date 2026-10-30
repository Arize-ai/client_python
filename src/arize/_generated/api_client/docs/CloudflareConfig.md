# CloudflareConfig

Config for a Cloudflare integration. Cloudflare integrations are used only by Decision Judge evaluators and cannot be used in prompts, experiments, or other features. Requests go to the Cloudflare account named by `account_id`, so there is no endpoint field, no custom request headers, and no model list. The API token is write-only and surfaces as `has_api_key`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**provider** | **str** | Discriminator identifying the Cloudflare provider. | 
**has_api_key** | **bool** | Whether an API token is configured (the token itself is never returned). | 
**account_id** | **str** | The 32-character Cloudflare account ID. | 

## Example

```python
from arize._generated.api_client.models.cloudflare_config import CloudflareConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CloudflareConfig from a JSON string
cloudflare_config_instance = CloudflareConfig.from_json(json)
# print the JSON string representation of the object
print(CloudflareConfig.to_json())

# convert the object into a dict
cloudflare_config_dict = cloudflare_config_instance.to_dict()
# create an instance of CloudflareConfig from a dict
cloudflare_config_from_dict = CloudflareConfig.from_dict(cloudflare_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


