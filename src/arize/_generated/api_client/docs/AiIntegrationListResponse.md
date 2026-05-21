# AiIntegrationListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**ai_integrations** | [**List[AiIntegration]**](AiIntegration.md) | A list of AI integrations | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.ai_integration_list_response import AiIntegrationListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of AiIntegrationListResponse from a JSON string
ai_integration_list_response_instance = AiIntegrationListResponse.from_json(json)
# print the JSON string representation of the object
print(AiIntegrationListResponse.to_json())

# convert the object into a dict
ai_integration_list_response_dict = ai_integration_list_response_instance.to_dict()
# create an instance of AiIntegrationListResponse from a dict
ai_integration_list_response_from_dict = AiIntegrationListResponse.from_dict(ai_integration_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


