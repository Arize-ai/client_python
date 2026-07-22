# UpdateAgentIntegrationRequest

Partial update body for `type=AGENT`. `type` is immutable; if present it must equal `AGENT` (422 otherwise). 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Discriminator. Immutable; must match the integration&#39;s type. | 
**name** | **str** |  | [optional] 
**description** | **str** |  | [optional] 
**scopings** | [**List[IntegrationScoping]**](IntegrationScoping.md) | Replace-on-provide. Empty array reverts to account-wide. | [optional] 
**config** | [**UpdateAgentConfig**](UpdateAgentConfig.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_agent_integration_request import UpdateAgentIntegrationRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateAgentIntegrationRequest from a JSON string
update_agent_integration_request_instance = UpdateAgentIntegrationRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateAgentIntegrationRequest.to_json())

# convert the object into a dict
update_agent_integration_request_dict = update_agent_integration_request_instance.to_dict()
# create an instance of UpdateAgentIntegrationRequest from a dict
update_agent_integration_request_from_dict = UpdateAgentIntegrationRequest.from_dict(update_agent_integration_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


