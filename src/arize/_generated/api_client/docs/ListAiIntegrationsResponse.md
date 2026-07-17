# ListAiIntegrationsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**ai_integrations** | [**List[AiIntegration]**](AiIntegration.md) | A list of AI integrations | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_ai_integrations_response import ListAiIntegrationsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListAiIntegrationsResponse from a JSON string
list_ai_integrations_response_instance = ListAiIntegrationsResponse.from_json(json)
# print the JSON string representation of the object
print(ListAiIntegrationsResponse.to_json())

# convert the object into a dict
list_ai_integrations_response_dict = list_ai_integrations_response_instance.to_dict()
# create an instance of ListAiIntegrationsResponse from a dict
list_ai_integrations_response_from_dict = ListAiIntegrationsResponse.from_dict(list_ai_integrations_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


