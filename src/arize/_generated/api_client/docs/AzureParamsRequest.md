# AzureParamsRequest

Azure OpenAI specific parameters in a write request (strict form of AzureParams)

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**azure_deployment_name** | **str** | The Azure deployment name | [optional] 
**azure_openai_endpoint** | **str** | The Azure OpenAI endpoint URL | [optional] 
**azure_openai_version** | **str** | The Azure OpenAI API version | [optional] 

## Example

```python
from arize._generated.api_client.models.azure_params_request import AzureParamsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AzureParamsRequest from a JSON string
azure_params_request_instance = AzureParamsRequest.from_json(json)
# print the JSON string representation of the object
print(AzureParamsRequest.to_json())

# convert the object into a dict
azure_params_request_dict = azure_params_request_instance.to_dict()
# create an instance of AzureParamsRequest from a dict
azure_params_request_from_dict = AzureParamsRequest.from_dict(azure_params_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


