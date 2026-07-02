# IntegrationBase

Fields shared by every integration, regardless of `type`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The integration ID (base64 global ID). | 
**type** | [**IntegrationType**](IntegrationType.md) |  | 
**name** | **str** | The integration name. Unique per (account, type). | 
**scopings** | [**List[IntegrationScoping]**](IntegrationScoping.md) | Visibility scoping rules. Account-wide when empty. | 
**created_at** | **datetime** | When the integration was created. | 
**updated_at** | **datetime** | When the integration was last updated. | 
**created_by_user_id** | **str** | Global ID of the user who created the integration. | 

## Example

```python
from arize._generated.api_client.models.integration_base import IntegrationBase

# TODO update the JSON string below
json = "{}"
# create an instance of IntegrationBase from a JSON string
integration_base_instance = IntegrationBase.from_json(json)
# print the JSON string representation of the object
print(IntegrationBase.to_json())

# convert the object into a dict
integration_base_dict = integration_base_instance.to_dict()
# create an instance of IntegrationBase from a dict
integration_base_from_dict = IntegrationBase.from_dict(integration_base_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


