# AiIntegrationScopingRequest

Visibility scoping for the integration in a write request (strict form of AiIntegrationScoping).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**organization_id** | **str** | Organization identifier (base64). Null means account-wide. | [optional] 
**space_id** | **str** | Space identifier (base64). Null means organization-wide (or account-wide if organization_id is also null). | [optional] 

## Example

```python
from arize._generated.api_client.models.ai_integration_scoping_request import AiIntegrationScopingRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AiIntegrationScopingRequest from a JSON string
ai_integration_scoping_request_instance = AiIntegrationScopingRequest.from_json(json)
# print the JSON string representation of the object
print(AiIntegrationScopingRequest.to_json())

# convert the object into a dict
ai_integration_scoping_request_dict = ai_integration_scoping_request_instance.to_dict()
# create an instance of AiIntegrationScopingRequest from a dict
ai_integration_scoping_request_from_dict = AiIntegrationScopingRequest.from_dict(ai_integration_scoping_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


