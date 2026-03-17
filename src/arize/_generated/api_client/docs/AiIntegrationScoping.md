# AiIntegrationScoping

Visibility scoping for the integration

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**organization_id** | **str** | Organization global ID. Null means account-wide. | [optional] 
**space_id** | **str** | Space global ID. Null means organization-wide (or account-wide if organization_id is also null). | [optional] 

## Example

```python
from arize._generated.api_client.models.ai_integration_scoping import AiIntegrationScoping

# TODO update the JSON string below
json = "{}"
# create an instance of AiIntegrationScoping from a JSON string
ai_integration_scoping_instance = AiIntegrationScoping.from_json(json)
# print the JSON string representation of the object
print(AiIntegrationScoping.to_json())

# convert the object into a dict
ai_integration_scoping_dict = ai_integration_scoping_instance.to_dict()
# create an instance of AiIntegrationScoping from a dict
ai_integration_scoping_from_dict = AiIntegrationScoping.from_dict(ai_integration_scoping_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


