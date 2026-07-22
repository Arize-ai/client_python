# AgentIntegration

An agent integration (type=AGENT): a customer-hosted HTTPS endpoint plus a JSON Schema describing the request payload. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the integration. | 
**type** | **str** | Discriminator identifying an agent integration. | 
**name** | **str** | The integration name. Unique per (account, type). | 
**description** | **str** | Optional human-readable description of the integration. | [optional] 
**scopings** | [**List[IntegrationScoping]**](IntegrationScoping.md) | Visibility scoping rules. Account-wide when empty. | 
**created_at** | **datetime** | When the integration was created. | 
**updated_at** | **datetime** | When the integration was last updated. | 
**created_by_user_id** | **str** | Unique identifier of the user who created the integration. Null if that user has since been deleted. | 
**config** | [**AgentConfig**](AgentConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.agent_integration import AgentIntegration

# TODO update the JSON string below
json = "{}"
# create an instance of AgentIntegration from a JSON string
agent_integration_instance = AgentIntegration.from_json(json)
# print the JSON string representation of the object
print(AgentIntegration.to_json())

# convert the object into a dict
agent_integration_dict = agent_integration_instance.to_dict()
# create an instance of AgentIntegration from a dict
agent_integration_from_dict = AgentIntegration.from_dict(agent_integration_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


