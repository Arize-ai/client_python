# IntegrationScoping

Visibility scoping for the integration.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**organization_id** | **str** | Organization identifier (base64). Null means account-wide. | [optional] 
**space_id** | **str** | Space identifier (base64). Null means organization-wide (or account-wide when organization_id is also null). | [optional] 

## Example

```python
from arize._generated.api_client.models.integration_scoping import IntegrationScoping

# TODO update the JSON string below
json = "{}"
# create an instance of IntegrationScoping from a JSON string
integration_scoping_instance = IntegrationScoping.from_json(json)
# print the JSON string representation of the object
print(IntegrationScoping.to_json())

# convert the object into a dict
integration_scoping_dict = integration_scoping_instance.to_dict()
# create an instance of IntegrationScoping from a dict
integration_scoping_from_dict = IntegrationScoping.from_dict(integration_scoping_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


