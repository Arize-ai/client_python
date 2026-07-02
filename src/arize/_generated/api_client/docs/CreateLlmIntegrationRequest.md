# CreateLlmIntegrationRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**name** | **str** | Integration name. Unique per (account, type). | 
**scopings** | [**List[IntegrationScoping]**](IntegrationScoping.md) | Visibility scoping rules. Defaults to account-wide. | [optional] 
**config** | [**CreateOpenAiConfig**](CreateOpenAiConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_llm_integration_request import CreateLlmIntegrationRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateLlmIntegrationRequest from a JSON string
create_llm_integration_request_instance = CreateLlmIntegrationRequest.from_json(json)
# print the JSON string representation of the object
print(CreateLlmIntegrationRequest.to_json())

# convert the object into a dict
create_llm_integration_request_dict = create_llm_integration_request_instance.to_dict()
# create an instance of CreateLlmIntegrationRequest from a dict
create_llm_integration_request_from_dict = CreateLlmIntegrationRequest.from_dict(create_llm_integration_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


