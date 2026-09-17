# CreateEvaluatorIntegrationRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Discriminator identifying this request as an evaluator integration. | 
**name** | **str** | Integration name. Must be unique among active AGENT and EVALUATOR integrations in the account. | 
**description** | **str** | Optional human-readable description of the integration. | [optional] 
**scopings** | [**List[IntegrationScopingRequest]**](IntegrationScopingRequest.md) | Visibility scoping rules. Defaults to account-wide if omitted or empty. A scoping with &#x60;space_id&#x60; set MUST also set &#x60;organization_id&#x60;.  | [optional] 
**config** | [**CreateEvaluatorIntegrationConfigInput**](CreateEvaluatorIntegrationConfigInput.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_evaluator_integration_request import CreateEvaluatorIntegrationRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateEvaluatorIntegrationRequest from a JSON string
create_evaluator_integration_request_instance = CreateEvaluatorIntegrationRequest.from_json(json)
# print the JSON string representation of the object
print(CreateEvaluatorIntegrationRequest.to_json())

# convert the object into a dict
create_evaluator_integration_request_dict = create_evaluator_integration_request_instance.to_dict()
# create an instance of CreateEvaluatorIntegrationRequest from a dict
create_evaluator_integration_request_from_dict = CreateEvaluatorIntegrationRequest.from_dict(create_evaluator_integration_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


