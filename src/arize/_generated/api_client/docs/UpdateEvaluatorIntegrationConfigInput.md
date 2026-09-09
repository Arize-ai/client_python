# UpdateEvaluatorIntegrationConfigInput

Partial evaluator config for PATCH. Omitted fields are left unchanged. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**endpoint** | **str** | New HTTPS endpoint URL. | [optional] 
**headers** | **Dict[str, str]** | Replace-on-provide. Pass &#x60;null&#x60; (or &#x60;{}&#x60;) to clear all headers. Encrypted at rest; never returned in responses.  | [optional] 
**input_schema** | **Dict[str, object]** | New JSON Schema for the request payload shape. The root schema must have &#x60;type: object&#x60;, must not define the reserved top-level &#x60;arize_metadata&#x60; field, and must not exceed 64 KiB.  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_evaluator_integration_config_input import UpdateEvaluatorIntegrationConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateEvaluatorIntegrationConfigInput from a JSON string
update_evaluator_integration_config_input_instance = UpdateEvaluatorIntegrationConfigInput.from_json(json)
# print the JSON string representation of the object
print(UpdateEvaluatorIntegrationConfigInput.to_json())

# convert the object into a dict
update_evaluator_integration_config_input_dict = update_evaluator_integration_config_input_instance.to_dict()
# create an instance of UpdateEvaluatorIntegrationConfigInput from a dict
update_evaluator_integration_config_input_from_dict = UpdateEvaluatorIntegrationConfigInput.from_dict(update_evaluator_integration_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


