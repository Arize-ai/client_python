# IntegrationScopingRequest

Visibility scoping for the integration in a write request (strict form of IntegrationScoping).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**organization_id** | **str** | Organization identifier (base64). Null means account-wide. | [optional] 
**space_id** | **str** | Space identifier (base64). Null means organization-wide (or account-wide when organization_id is also null). | [optional] 

## Example

```python
from arize._generated.api_client.models.integration_scoping_request import IntegrationScopingRequest

# TODO update the JSON string below
json = "{}"
# create an instance of IntegrationScopingRequest from a JSON string
integration_scoping_request_instance = IntegrationScopingRequest.from_json(json)
# print the JSON string representation of the object
print(IntegrationScopingRequest.to_json())

# convert the object into a dict
integration_scoping_request_dict = integration_scoping_request_instance.to_dict()
# create an instance of IntegrationScopingRequest from a dict
integration_scoping_request_from_dict = IntegrationScopingRequest.from_dict(integration_scoping_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


