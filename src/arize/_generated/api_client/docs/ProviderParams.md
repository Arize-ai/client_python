# ProviderParams

Provider-specific parameters

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**azure_params** | [**ProviderParamsAzureParams**](ProviderParamsAzureParams.md) |  | [optional] 
**anthropic_headers** | [**ProviderParamsAnthropicHeaders**](ProviderParamsAnthropicHeaders.md) |  | [optional] 
**anthropic_version** | **str** | Anthropic API version | [optional] 
**bedrock_options** | [**ProviderParamsBedrockOptions**](ProviderParamsBedrockOptions.md) |  | [optional] 
**custom_provider_params** | **object** | Custom provider parameters | [optional] 
**region** | **str** | Region for the model deployment | [optional] 

## Example

```python
from arize._generated.api_client.models.provider_params import ProviderParams

# TODO update the JSON string below
json = "{}"
# create an instance of ProviderParams from a JSON string
provider_params_instance = ProviderParams.from_json(json)
# print the JSON string representation of the object
print(ProviderParams.to_json())

# convert the object into a dict
provider_params_dict = provider_params_instance.to_dict()
# create an instance of ProviderParams from a dict
provider_params_from_dict = ProviderParams.from_dict(provider_params_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


