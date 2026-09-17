# UpdateIntegrationRequest

Partial update of an integration, discriminated by `type` (immutable). The `type` field selects the per-type PATCH shape. Provide at least one updatable field in addition to `type`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Discriminator. Immutable; must match the integration&#39;s type. | 
**name** | **str** | New integration name. Must be unique among active AGENT and EVALUATOR integrations in the account. | [optional] 
**scopings** | [**List[IntegrationScopingRequest]**](IntegrationScopingRequest.md) | Replace-on-provide. Empty array reverts to account-wide. | [optional] 
**config** | [**UpdateEvaluatorIntegrationConfigInput**](UpdateEvaluatorIntegrationConfigInput.md) |  | [optional] 
**description** | **str** | New human-readable description of the integration. Pass null to clear it. | [optional] 

## Example

```python
from arize._generated.api_client.models.update_integration_request import UpdateIntegrationRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateIntegrationRequest from a JSON string
update_integration_request_instance = UpdateIntegrationRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateIntegrationRequest.to_json())

# convert the object into a dict
update_integration_request_dict = update_integration_request_instance.to_dict()
# create an instance of UpdateIntegrationRequest from a dict
update_integration_request_from_dict = UpdateIntegrationRequest.from_dict(update_integration_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


