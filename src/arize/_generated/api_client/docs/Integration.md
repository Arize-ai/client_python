# Integration

A polymorphic integration resource. The `type` field selects the `config` shape; for `LLM`, `config.provider` selects the per-provider config.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the integration. | 
**type** | **str** | Discriminator identifying an LLM integration. | 
**name** | **str** | The integration name. Unique per (account, type). | 
**scopings** | [**List[IntegrationScoping]**](IntegrationScoping.md) | Visibility scoping rules. Account-wide when empty. | 
**created_at** | **datetime** | When the integration was created. | 
**updated_at** | **datetime** | When the integration was last updated. | 
**created_by_user_id** | **str** | Unique identifier of the user who created the integration. Null if that user has since been deleted. | 
**config** | [**AgentConfig**](AgentConfig.md) |  | 
**description** | **str** | Optional human-readable description of the integration. | [optional] 

## Example

```python
from arize._generated.api_client.models.integration import Integration

# TODO update the JSON string below
json = "{}"
# create an instance of Integration from a JSON string
integration_instance = Integration.from_json(json)
# print the JSON string representation of the object
print(Integration.to_json())

# convert the object into a dict
integration_dict = integration_instance.to_dict()
# create an instance of Integration from a dict
integration_from_dict = Integration.from_dict(integration_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


