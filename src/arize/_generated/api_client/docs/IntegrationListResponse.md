# IntegrationListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**integrations** | [**List[LlmIntegration]**](LlmIntegration.md) | A polymorphic, type-tagged list of integrations. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.integration_list_response import IntegrationListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of IntegrationListResponse from a JSON string
integration_list_response_instance = IntegrationListResponse.from_json(json)
# print the JSON string representation of the object
print(IntegrationListResponse.to_json())

# convert the object into a dict
integration_list_response_dict = integration_list_response_instance.to_dict()
# create an instance of IntegrationListResponse from a dict
integration_list_response_from_dict = IntegrationListResponse.from_dict(integration_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


