# CreateAgentIntegrationRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**name** | **str** | Integration name (unique within the account). | 
**description** | **str** |  | [optional] 
**scopings** | [**List[IntegrationScoping]**](IntegrationScoping.md) | Visibility scoping rules. Defaults to account-wide if omitted or empty. A scoping with &#x60;space_id&#x60; set MUST also set &#x60;organization_id&#x60;.  | [optional] 
**config** | [**CreateAgentConfig**](CreateAgentConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_agent_integration_request import CreateAgentIntegrationRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAgentIntegrationRequest from a JSON string
create_agent_integration_request_instance = CreateAgentIntegrationRequest.from_json(json)
# print the JSON string representation of the object
print(CreateAgentIntegrationRequest.to_json())

# convert the object into a dict
create_agent_integration_request_dict = create_agent_integration_request_instance.to_dict()
# create an instance of CreateAgentIntegrationRequest from a dict
create_agent_integration_request_from_dict = CreateAgentIntegrationRequest.from_dict(create_agent_integration_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


