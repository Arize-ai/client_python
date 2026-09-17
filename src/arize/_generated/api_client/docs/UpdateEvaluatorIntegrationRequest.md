# UpdateEvaluatorIntegrationRequest

Partial update body for `type=EVALUATOR`. `type` is immutable; if present it must equal `EVALUATOR` (422 otherwise). 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Discriminator. Immutable; must match the integration&#39;s type. | 
**name** | **str** | New integration name. Must be unique among active AGENT and EVALUATOR integrations in the account. | [optional] 
**description** | **str** | New human-readable description of the integration. Pass null to clear it. | [optional] 
**scopings** | [**List[IntegrationScopingRequest]**](IntegrationScopingRequest.md) | Replace-on-provide. Empty array reverts to account-wide. | [optional] 
**config** | [**UpdateEvaluatorIntegrationConfigInput**](UpdateEvaluatorIntegrationConfigInput.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_evaluator_integration_request import UpdateEvaluatorIntegrationRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateEvaluatorIntegrationRequest from a JSON string
update_evaluator_integration_request_instance = UpdateEvaluatorIntegrationRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateEvaluatorIntegrationRequest.to_json())

# convert the object into a dict
update_evaluator_integration_request_dict = update_evaluator_integration_request_instance.to_dict()
# create an instance of UpdateEvaluatorIntegrationRequest from a dict
update_evaluator_integration_request_from_dict = UpdateEvaluatorIntegrationRequest.from_dict(update_evaluator_integration_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


