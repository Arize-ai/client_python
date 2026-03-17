# AiIntegrationsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**ai_integrations** | [**List[AiIntegration]**](AiIntegration.md) | A list of AI integrations | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.ai_integrations_list200_response import AiIntegrationsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of AiIntegrationsList200Response from a JSON string
ai_integrations_list200_response_instance = AiIntegrationsList200Response.from_json(json)
# print the JSON string representation of the object
print(AiIntegrationsList200Response.to_json())

# convert the object into a dict
ai_integrations_list200_response_dict = ai_integrations_list200_response_instance.to_dict()
# create an instance of AiIntegrationsList200Response from a dict
ai_integrations_list200_response_from_dict = AiIntegrationsList200Response.from_dict(ai_integrations_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


