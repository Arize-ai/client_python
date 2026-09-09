# EvaluatorIntegration

An evaluator integration (type=EVALUATOR): a customer-hosted HTTPS endpoint plus a JSON Schema describing the request payload. Used to run remote evaluators against LLM outputs. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the integration. | 
**type** | **str** | Discriminator identifying an evaluator integration. | 
**name** | **str** | The integration name. Unique among active AGENT and EVALUATOR integrations in the account. | 
**description** | **str** | Optional human-readable description of the integration. | [optional] 
**scopings** | [**List[IntegrationScoping]**](IntegrationScoping.md) | Visibility scoping rules. Account-wide when empty. | 
**created_at** | **datetime** | When the integration was created. | 
**updated_at** | **datetime** | When the integration was last updated. | 
**created_by_user_id** | **str** | Unique identifier of the user who created the integration. Null if that user has since been deleted. | 
**config** | [**EvaluatorIntegrationConfig**](EvaluatorIntegrationConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.evaluator_integration import EvaluatorIntegration

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorIntegration from a JSON string
evaluator_integration_instance = EvaluatorIntegration.from_json(json)
# print the JSON string representation of the object
print(EvaluatorIntegration.to_json())

# convert the object into a dict
evaluator_integration_dict = evaluator_integration_instance.to_dict()
# create an instance of EvaluatorIntegration from a dict
evaluator_integration_from_dict = EvaluatorIntegration.from_dict(evaluator_integration_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


