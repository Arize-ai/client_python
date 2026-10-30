# CreateCloudflareConfig

Create config for a Cloudflare integration, used only by Decision Judge evaluators. `api_key` (a Cloudflare API token) and `account_id` are required. The token is write-only (never returned; it surfaces as `has_api_key` on read). There is no endpoint field, no custom request headers, and no model list. Creating one returns `400` when this deployment has not enabled the Cloudflare provider. This is a deployment setting, separate from the account's Decision Judge access that the evaluator and task endpoints report as `403`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**provider** | **str** | Discriminator identifying the Cloudflare provider. | 
**api_key** | **str** | Cloudflare API token (write-only, never returned). | 
**account_id** | **str** | The 32-character Cloudflare account ID. | 

## Example

```python
from arize._generated.api_client.models.create_cloudflare_config import CreateCloudflareConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateCloudflareConfig from a JSON string
create_cloudflare_config_instance = CreateCloudflareConfig.from_json(json)
# print the JSON string representation of the object
print(CreateCloudflareConfig.to_json())

# convert the object into a dict
create_cloudflare_config_dict = create_cloudflare_config_instance.to_dict()
# create an instance of CreateCloudflareConfig from a dict
create_cloudflare_config_from_dict = CreateCloudflareConfig.from_dict(create_cloudflare_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


