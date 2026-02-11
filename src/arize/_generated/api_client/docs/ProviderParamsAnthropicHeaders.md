# ProviderParamsAnthropicHeaders

Anthropic-specific headers

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**anthropic_beta** | **List[Optional[str]]** | Anthropic beta feature flags | [optional] 

## Example

```python
from arize._generated.api_client.models.provider_params_anthropic_headers import ProviderParamsAnthropicHeaders

# TODO update the JSON string below
json = "{}"
# create an instance of ProviderParamsAnthropicHeaders from a JSON string
provider_params_anthropic_headers_instance = ProviderParamsAnthropicHeaders.from_json(json)
# print the JSON string representation of the object
print(ProviderParamsAnthropicHeaders.to_json())

# convert the object into a dict
provider_params_anthropic_headers_dict = provider_params_anthropic_headers_instance.to_dict()
# create an instance of ProviderParamsAnthropicHeaders from a dict
provider_params_anthropic_headers_from_dict = ProviderParamsAnthropicHeaders.from_dict(provider_params_anthropic_headers_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


