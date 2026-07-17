# UpdateLlmIntegrationRequest

PATCH body for an `LLM` integration. `type` is required (it selects the union member) and immutable. Provide at least one updatable field (`name`, `scopings`, or `config`) in addition to `type`. `scopings` replaces on provide.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Discriminator. Immutable; must match the integration&#39;s type. | 
**name** | **str** | New integration name. | [optional] 
**scopings** | [**List[IntegrationScoping]**](IntegrationScoping.md) | Replaces the existing scoping rules. | [optional] 
**config** | [**UpdateLlmConfig**](UpdateLlmConfig.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_llm_integration_request import UpdateLlmIntegrationRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateLlmIntegrationRequest from a JSON string
update_llm_integration_request_instance = UpdateLlmIntegrationRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateLlmIntegrationRequest.to_json())

# convert the object into a dict
update_llm_integration_request_dict = update_llm_integration_request_instance.to_dict()
# create an instance of UpdateLlmIntegrationRequest from a dict
update_llm_integration_request_from_dict = UpdateLlmIntegrationRequest.from_dict(update_llm_integration_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


