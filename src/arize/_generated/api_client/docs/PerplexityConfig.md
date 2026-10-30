# PerplexityConfig

Config for a Perplexity integration. Perplexity integrations are used only by Decision Judge evaluators and cannot be used in prompts, experiments, or other features. Perplexity is a single hosted service, so there is no endpoint field, no custom request headers, and no model list. The API key is write-only and surfaces as `has_api_key`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**provider** | **str** | Discriminator identifying the Perplexity provider. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 

## Example

```python
from arize._generated.api_client.models.perplexity_config import PerplexityConfig

# TODO update the JSON string below
json = "{}"
# create an instance of PerplexityConfig from a JSON string
perplexity_config_instance = PerplexityConfig.from_json(json)
# print the JSON string representation of the object
print(PerplexityConfig.to_json())

# convert the object into a dict
perplexity_config_dict = perplexity_config_instance.to_dict()
# create an instance of PerplexityConfig from a dict
perplexity_config_from_dict = PerplexityConfig.from_dict(perplexity_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


