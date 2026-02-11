# ProviderParamsBedrockOptions

AWS Bedrock options

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**use_converse_endpoint** | **bool** | Whether to use the AWS Bedrock Converse endpoint | [optional] 

## Example

```python
from arize._generated.api_client.models.provider_params_bedrock_options import ProviderParamsBedrockOptions

# TODO update the JSON string below
json = "{}"
# create an instance of ProviderParamsBedrockOptions from a JSON string
provider_params_bedrock_options_instance = ProviderParamsBedrockOptions.from_json(json)
# print the JSON string representation of the object
print(ProviderParamsBedrockOptions.to_json())

# convert the object into a dict
provider_params_bedrock_options_dict = provider_params_bedrock_options_instance.to_dict()
# create an instance of ProviderParamsBedrockOptions from a dict
provider_params_bedrock_options_from_dict = ProviderParamsBedrockOptions.from_dict(provider_params_bedrock_options_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


