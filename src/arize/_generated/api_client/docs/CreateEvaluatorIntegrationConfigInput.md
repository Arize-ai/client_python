# CreateEvaluatorIntegrationConfigInput

Write shape for the evaluator config on create. `headers` is encrypted at rest and never returned in responses; reads surface `has_headers`. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**endpoint** | **str** | HTTPS endpoint requests are sent to. Validated server-side and must resolve to a public address. | 
**headers** | **Dict[str, str]** | Cleartext header map. Encrypted at rest; never returned in responses. Omitting this field on create means no headers are configured.  | [optional] 
**input_schema** | **Dict[str, object]** | JSON Schema (Draft-07) the endpoint&#39;s request body conforms to. The root schema must have &#x60;type: object&#x60;, must not define the reserved top-level &#x60;arize_metadata&#x60; field, and must not exceed 64 KiB.  | 

## Example

```python
from arize._generated.api_client.models.create_evaluator_integration_config_input import CreateEvaluatorIntegrationConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of CreateEvaluatorIntegrationConfigInput from a JSON string
create_evaluator_integration_config_input_instance = CreateEvaluatorIntegrationConfigInput.from_json(json)
# print the JSON string representation of the object
print(CreateEvaluatorIntegrationConfigInput.to_json())

# convert the object into a dict
create_evaluator_integration_config_input_dict = create_evaluator_integration_config_input_instance.to_dict()
# create an instance of CreateEvaluatorIntegrationConfigInput from a dict
create_evaluator_integration_config_input_from_dict = CreateEvaluatorIntegrationConfigInput.from_dict(create_evaluator_integration_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


