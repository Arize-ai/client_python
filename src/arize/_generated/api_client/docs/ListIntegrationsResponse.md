# ListIntegrationsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**integrations** | [**List[LlmIntegration]**](LlmIntegration.md) | A polymorphic, type-tagged list of integrations. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_integrations_response import ListIntegrationsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListIntegrationsResponse from a JSON string
list_integrations_response_instance = ListIntegrationsResponse.from_json(json)
# print the JSON string representation of the object
print(ListIntegrationsResponse.to_json())

# convert the object into a dict
list_integrations_response_dict = list_integrations_response_instance.to_dict()
# create an instance of ListIntegrationsResponse from a dict
list_integrations_response_from_dict = ListIntegrationsResponse.from_dict(list_integrations_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


