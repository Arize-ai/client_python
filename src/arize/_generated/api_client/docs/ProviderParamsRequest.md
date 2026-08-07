# ProviderParamsRequest

Provider-specific parameters in a write request (strict form of ProviderParams; leaf schemas use *Request variants)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**azure_params** | [**AzureParamsRequest**](AzureParamsRequest.md) | Azure OpenAI specific parameters | [optional] 
**anthropic_headers** | [**AnthropicHeadersRequest**](AnthropicHeadersRequest.md) | Anthropic-specific headers | [optional] 
**anthropic_version** | **str** | Anthropic API version | [optional] 
**bedrock_options** | [**BedrockOptionsRequest**](BedrockOptionsRequest.md) | AWS Bedrock options | [optional] 
**region** | **str** | Region for the model deployment | [optional] 

## Example

```python
from arize._generated.api_client.models.provider_params_request import ProviderParamsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ProviderParamsRequest from a JSON string
provider_params_request_instance = ProviderParamsRequest.from_json(json)
# print the JSON string representation of the object
print(ProviderParamsRequest.to_json())

# convert the object into a dict
provider_params_request_dict = provider_params_request_instance.to_dict()
# create an instance of ProviderParamsRequest from a dict
provider_params_request_from_dict = ProviderParamsRequest.from_dict(provider_params_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


