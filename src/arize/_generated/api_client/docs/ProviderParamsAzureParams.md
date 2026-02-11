# ProviderParamsAzureParams

Azure OpenAI specific parameters

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**azure_deployment_name** | **str** | The Azure deployment name | [optional] 
**azure_openai_endpoint** | **str** | The Azure OpenAI endpoint URL | [optional] 
**azure_openai_version** | **str** | The Azure OpenAI API version | [optional] 

## Example

```python
from arize._generated.api_client.models.provider_params_azure_params import ProviderParamsAzureParams

# TODO update the JSON string below
json = "{}"
# create an instance of ProviderParamsAzureParams from a JSON string
provider_params_azure_params_instance = ProviderParamsAzureParams.from_json(json)
# print the JSON string representation of the object
print(ProviderParamsAzureParams.to_json())

# convert the object into a dict
provider_params_azure_params_dict = provider_params_azure_params_instance.to_dict()
# create an instance of ProviderParamsAzureParams from a dict
provider_params_azure_params_from_dict = ProviderParamsAzureParams.from_dict(provider_params_azure_params_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


