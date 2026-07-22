# CreateIntegrationRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**name** | **str** | Integration name (unique within the account). | 
**scopings** | [**List[IntegrationScoping]**](IntegrationScoping.md) | Visibility scoping rules. Defaults to account-wide if omitted or empty. A scoping with &#x60;space_id&#x60; set MUST also set &#x60;organization_id&#x60;.  | [optional] 
**config** | [**CreateAgentConfig**](CreateAgentConfig.md) |  | 
**description** | **str** |  | [optional] 

## Example

```python
from arize._generated.api_client.models.create_integration_request import CreateIntegrationRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateIntegrationRequest from a JSON string
create_integration_request_instance = CreateIntegrationRequest.from_json(json)
# print the JSON string representation of the object
print(CreateIntegrationRequest.to_json())

# convert the object into a dict
create_integration_request_dict = create_integration_request_instance.to_dict()
# create an instance of CreateIntegrationRequest from a dict
create_integration_request_from_dict = CreateIntegrationRequest.from_dict(create_integration_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


